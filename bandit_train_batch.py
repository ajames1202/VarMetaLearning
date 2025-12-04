import os
import argparse
import time
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import ray

import visual_bandit_env2 as vbe
import var_bandit_learner2 as bl

def extract_pair_view(obs_tensor, env, crop_size=112, pad=6, mask_cursor=False):
    """
    obs_tensor: (1,3,H,W) in [0,1]
    returns: (1,3,crop_size, 2*crop_size) = [left_crop | right_crop]
    """
    lr = env.unwrapped.left_rect
    rr = env.unwrapped.right_rect
    _, _, H, W = obs_tensor.shape

    # Shared vertical crop to keep left/right same height
    y1 = max(min(lr.top, rr.top) - pad, 0)
    y2 = min(max(lr.bottom, rr.bottom) + pad, H)

    def crop_resize(rect):
        x1 = max(rect.left - pad, 0)
        x2 = min(rect.right + pad, W)
        patch = obs_tensor[:, :, y1:y2, x1:x2]  # (1,3,h,w)
        patch = F.interpolate(patch, size=(crop_size, crop_size),
                              mode="bilinear", align_corners=False)
        return patch

    left  = crop_resize(lr)
    right = crop_resize(rr)

    pair = torch.cat([left, right], dim=-1)  # width concat

    if mask_cursor:
        cx, cy = env.unwrapped.cursor
        cx, cy = int(cx), int(cy)
        # cursor mask in ORIGINAL coords → skip (cursor is usually outside crops at trial start),
        # but you can instead mask inside 'pair' if you know it can overlap the stimuli.
        pass

    return pair

# -------------------------
# Utilities
# -------------------------
def save_checkpoint(path: str, agent: torch.nn.Module, optim_bandit, optim_motor, extra: Dict[str, Any] | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model_state": {k: v.detach().cpu() for k, v in agent.state_dict().items()},
        "optim_bandit": optim_bandit.state_dict() if optim_bandit is not None else None,
        "optim_motor": optim_motor.state_dict() if optim_motor is not None else None,
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, agent: torch.nn.Module, optim_bandit=None, optim_motor=None, map_location="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    agent.load_state_dict(ckpt["model_state"], strict=True)
    if optim_bandit is not None and ckpt.get("optim_bandit") is not None:
        optim_bandit.load_state_dict(ckpt["optim_bandit"])
    if optim_motor is not None and ckpt.get("optim_motor") is not None:
        optim_motor.load_state_dict(ckpt["optim_motor"])
    return ckpt.get("extra", {})


# -------------------------
# Rollout with Query + Memory tokens
# -------------------------
@torch.inference_mode()
def meta_ep_rollout(env, agent, device, session_K: int, session_N: int, worker_id: int = 0, print_this_session: bool = False):
    
    def log(*args, **kwargs):
        if print_this_session and worker_id == 0:
            print(*args, **kwargs)

    
    agent.eval()

    history = torch.empty((0, agent.input_size), dtype=torch.float32, device=device)

    obs, info = env.reset()
    done = False

    high_reward_choice_per_N = np.zeros(session_N, dtype=np.float32)
    cum_high_reward_choice = 0.0


    # ---------- precompute constants ----------
    W, H = env.unwrapped.W, env.unwrapped.H
    sx = 2.0 / (W - 1)
    sy = 2.0 / (H - 1)
    ox = -1.0
    oy = -1.0

    left_c = np.array([env.unwrapped.left_rect.centerx,  env.unwrapped.left_rect.centery],  np.float32)
    right_c= np.array([env.unwrapped.right_rect.centerx, env.unwrapped.right_rect.centery], np.float32)
    left_c_norm  = np.array([left_c[0]  * sx + ox, left_c[1]  * sy + oy], np.float32)
    right_c_norm = np.array([right_c[0] * sx + ox, right_c[1] * sy + oy], np.float32)

    # ---------- reusable tensors (avoid allocs) ----------
    zero_act = torch.zeros((1, 2), device=device, dtype=torch.float32)
    zero_r   = torch.zeros((1, 1), device=device, dtype=torch.float32)
    meta_ep_start_torch = torch.zeros((1, 1), device=device, dtype=torch.float32)

    xy_pos_t  = torch.empty((1, 2), device=device, dtype=torch.float32)
    goal_vec_t= torch.empty((1, 2), device=device, dtype=torch.float32)

    # Buffers
    xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf = [], [], []
    obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf = [], [], [], []

    meta_trial_idx = 0
    ep_start_flag = 1.0

    trial_feat = None
    trial_action = None
    trial_action_np = None
    trial_action_idx = 0
    trial_small_np = None
    trial_meta_start = 0.0

    while not done:

        # -----------------------
        # TRIAL START: compute vision ONCE + choose arm
        # -----------------------
        if ep_start_flag == 1.0:
            trial_meta_start = 1.0 if meta_trial_idx == 0 else 0.0
            meta_ep_start_torch.fill_(trial_meta_start)

            # only now convert obs -> torch and run CNN
            obs_tensor = torch.from_numpy(obs).to(device).permute(2, 0, 1).unsqueeze(0).float()
            obs_tensor.mul_(1.0 / 255.0)

            # small = agent.downsample(obs_tensor)     # (1,C,h,w)
            pair_view = extract_pair_view(obs_tensor, env, crop_size=112, pad=6)  # (1,3,112,224)
            f1 = agent.encode(pair_view)                 # (1,F)
            trial_feat = f1

            q_out, _, history = agent.rnn_fwd(f1, zero_act, zero_r, meta_ep_start_torch, history=history)
            reward_logits = agent.reward_compute(q_out.unsqueeze(0), f1).squeeze(0)
            probs = torch.sigmoid(reward_logits).clamp(1e-4, 1.0 - 1e-4)

            # Thompson sampling (same logic)
            concentration = 5.0
            alpha = probs * concentration + 1.0
            beta  = (1.0 - probs) * concentration + 1.0
            u = torch.stack([
                torch.distributions.Beta(alpha[0], beta[0]).rsample(),
                torch.distributions.Beta(alpha[1], beta[1]).rsample()
            ])
            a_t = torch.argmax(u)

            trial_action_idx = int(a_t.item())
            trial_action = F.one_hot(a_t, num_classes=2).unsqueeze(0).float()

            # cache for trial end + motor loop
            trial_action_np = trial_action.squeeze(0).cpu().numpy()  # one time per trial
            trial_feat_np = f1.squeeze(0).cpu().numpy()          # one time per trial

        # -----------------------
        # MOTOR step
        # -----------------------
        x_pix, y_pix = env.unwrapped.cursor
        x_norm = x_pix * sx + ox
        y_norm = y_pix * sy + oy

        goal_center = left_c_norm if trial_action_idx == 0 else right_c_norm
        g0 = goal_center[0] - x_norm
        g1 = goal_center[1] - y_norm

        # write into preallocated tensors
        xy_pos_t[0, 0] = float(x_norm)
        xy_pos_t[0, 1] = float(y_norm)
        goal_vec_t[0, 0] = float(g0)
        goal_vec_t[0, 1] = float(g1)


        mu, log_std = agent.motor_fwd(trial_action, xy_pos=xy_pos_t, goal_vec=goal_vec_t)
        std = torch.exp(log_std)
        y = mu + std * torch.randn_like(std)
        action = torch.tanh(y)
        action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, term, trunc, info = env.step(action_np)
        done = bool(term) or bool(trunc)

        # save motor buffers (already in numpy)
        xy_pos_buf.append(np.array([x_norm, y_norm], np.float32))
        goal_vec_buf.append(np.array([g0, g1], np.float32))
        chosen_bandits_motor_buf.append(trial_action_np)

        # -----------------------
        # TRIAL END
        # -----------------------
        if info.get("trial_ended", False):
            r_t = torch.tensor([[float(reward)]], device=device, dtype=torch.float32)
            meta_ep_start_torch.fill_(trial_meta_start)

            _, _, history = agent.rnn_fwd(trial_feat, trial_action, r_t, meta_ep_start_torch, history=history)

            obs_bandit.append(trial_feat_np)
            chosen_bandits_buf.append(trial_action_np)
            bandit_rewards_buf.append(float(reward))
            meta_ep_start_buf.append(float(trial_meta_start))
            # cum_high_reward_choice += float(reward)
            selected_high_reward = info.get("selected_high_reward_this_trial", -1)
            if selected_high_reward:
                idxN = min(int(info.get("trial_index_in_pair", 0)), session_N - 1)
                high_reward_choice_per_N[idxN] += 1.0
                cum_high_reward_choice += 1

            reached_target = info.get("selected_target")
            curr_pair_index = info.get("pair_index_in_session", -1)
            log(
                f"[W{worker_id}] trial={meta_trial_idx} reward={reward}, "
                f"pair_idx={curr_pair_index}, "
                f"choice={int(trial_action.argmax(dim=-1).item())}, "
                f"reached_target={reached_target}, "
                f"selected_high_reward={selected_high_reward}, "
                f"probs={probs} "
            )

            meta_trial_idx += 1
            ep_start_flag = 1.0
        else:
            ep_start_flag = 0.0

        obs = next_obs

    return {
        "xy_pos_buf": xy_pos_buf,
        "goal_vec_buf": goal_vec_buf,
        "chosen_bandits_motor_buf": chosen_bandits_motor_buf,
        "obs_bandit": obs_bandit,
        "chosen_bandits_buf": chosen_bandits_buf,
        "bandit_rewards_buf": bandit_rewards_buf,
        "meta_ep_start_buf": meta_ep_start_buf,
        "metrics": {    "cum_high_reward_choice": float(cum_high_reward_choice),
            "high_reward_choice_per_N": high_reward_choice_per_N,
            "num_trials": int(len(bandit_rewards_buf)),
        }
    }

# -------------------------
# Ray worker
# -------------------------
@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    def __init__(self, session_K: int, session_N: int, seed: int = 0, worker_id: int = 0):
        self.session_K = session_K
        self.session_N = session_N
        self.worker_id = worker_id
        self.device = torch.device("cpu")

        self.env = vbe.TwoChoiceReachingEnv(
            W=384,
            H=400,
            render_mode="rgb_array",
            seed=seed + 1000 * worker_id,
            session_K=session_K,
            session_N=session_N,
            randomize_sides=True
        )

        # Agent copy (CPU). Must match driver hyperparams.
        feature_dim = 64
        input_size = feature_dim + 2 + 1 + 1
        hidden = 128
        action_dim = 2

        self.agent = bl.BanditLearner(
            input_size=input_size,
            feature_dim=feature_dim,
            rnn_hidden_size=hidden,
            action_dim=action_dim,
        ).to(self.device)

    def rollout(self, agent_state_cpu: Dict[str, torch.Tensor], print_this_session: bool = False) -> Dict[str, Any]:
        # Load params from driver
        self.agent.load_state_dict(agent_state_cpu, strict=True)
        return meta_ep_rollout(
            env=self.env,
            agent=self.agent,
            device=self.device,
            session_K=self.session_K,
            session_N=self.session_N,
            worker_id=self.worker_id,
            print_this_session=print_this_session,
        )


# -------------------------
# Main training loop
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--episodes-per-update", type=int, default=8)  # B
    parser.add_argument("--num-updates", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=50)

    # env/session params
    parser.add_argument("--session-K", type=int, default=3)
    parser.add_argument("--session-N", type=int, default=12)

    # model params (keep consistent with worker)
    parser.add_argument("--feature-dim", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Ray init
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Device for training
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build train agent
    input_size = args.feature_dim + 2 + 1 + 1
    action_dim = 2

    agent = bl.BanditLearner(
        input_size=input_size,
        feature_dim=args.feature_dim,
        rnn_hidden_size=args.hidden,
        action_dim=action_dim,
    ).to(train_device)

    optim_bandit = torch.optim.Adam(agent.bandit_parameters(), lr=3e-4)
    optim_motor = torch.optim.Adam(agent.motor_parameters(), lr=3e-4)

    # Workers
    workers = [
        RolloutWorker.remote(args.session_K, args.session_N, seed=args.seed, worker_id=i)
        for i in range(args.num_workers)
    ]

    # TensorBoard
    log_dir = os.path.join("runs", f"bandit_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)

    global_step = 0

    for upd in range(args.num_updates):
        # Broadcast state dict (CPU tensors) to workers
        agent_state_cpu = {k: v.detach().cpu() for k, v in agent.state_dict().items()}
        state_ref = ray.put(agent_state_cpu)

        # Launch rollouts
        num_rollouts = args.episodes_per_update
        futures = []
        for i in range(num_rollouts):
            w = workers[i % len(workers)]
            if(upd % 10 == 0 and i % len(workers) == 0):
                print_this_session = True
            else:
                print_this_session = False 

            # print("upd=",upd, ", w=", i % len(workers), ", print_this_session=", print_this_session)       
            futures.append(w.rollout.remote(state_ref, print_this_session=print_this_session))

        results: List[Dict[str, Any]] = ray.get(futures)

        # Aggregate buffers
        xy_pos_buf = []
        goal_vec_buf = []
        chosen_bandits_motor_buf = []

        # IMPORTANT: keep EPISODES as batch items
        obs_bandit_batch = []          # list of (T,C,H,W)
        chosen_bandits_batch = []      # list of (T,2)
        bandit_rewards_batch = []      # list of (T,)
        meta_ep_start_batch = []       # list of (T,)

        cum_high_reward_choice = 0.0
        total_trials = 0



        for r in results:
            # motor buffers can stay flattened
            xy_pos_buf.extend(r["xy_pos_buf"])
            goal_vec_buf.extend(r["goal_vec_buf"])
            chosen_bandits_motor_buf.extend(r["chosen_bandits_motor_buf"])

            # bandit buffers: make each rollout an episode entry
            T = len(r["bandit_rewards_buf"])
            if T == 0:
                continue

            obs_ep = np.stack(r["obs_bandit"], axis=0)                 # (T,F)
            a_ep   = np.stack(r["chosen_bandits_buf"], axis=0)         # (T,2)
            r_ep   = np.asarray(r["bandit_rewards_buf"], dtype=np.float32)  # (T,)
            s_ep   = np.asarray(r["meta_ep_start_buf"], dtype=np.float32)   # (T,)

            obs_bandit_batch.append(obs_ep)
            chosen_bandits_batch.append(a_ep)
            bandit_rewards_batch.append(r_ep)
            meta_ep_start_batch.append(s_ep)

            cum_high_reward_choice += r["metrics"]["cum_high_reward_choice"]
            total_trials += r["metrics"]["num_trials"]

        cum_high_reward_choice /= args.episodes_per_update

        # OPTIONAL safety: ensure all episodes same T (update2 assumes this)
        T0 = obs_bandit_batch[0].shape[0]
        obs_bandit_batch = [x for x in obs_bandit_batch if x.shape[0] == T0]
        chosen_bandits_batch = chosen_bandits_batch[:len(obs_bandit_batch)]
        bandit_rewards_batch = bandit_rewards_batch[:len(obs_bandit_batch)]
        meta_ep_start_batch = meta_ep_start_batch[:len(obs_bandit_batch)]

        # Train update
        agent.train()
        var_loss, motor_loss = agent.update2(
            optim_bandit=optim_bandit,
            optim_motor=optim_motor,
            xy_pos_buf=xy_pos_buf,
            goal_vec_buf=goal_vec_buf,
            chosen_bandits_motor_buf=chosen_bandits_motor_buf,
            bandit_obs=obs_bandit_batch,                 # <-- changed
            chosen_bandits_buf=chosen_bandits_batch,     # <-- changed
            bandit_rewards_buf=bandit_rewards_batch,     # <-- changed
            meta_ep_start_buf=meta_ep_start_batch,       # <-- changed
            device=train_device,
        )

        # Logs
        writer.add_scalar("loss/variational", float(var_loss), upd)
        writer.add_scalar("loss/motor", float(motor_loss), upd)
        writer.add_scalar("rollout/cum_high_reward_choice", float(cum_high_reward_choice), upd)
        writer.add_scalar("rollout/total_trials", int(total_trials), upd)

        if upd % 10 == 0:
            print(f"[upd {upd:04d}] var_loss={var_loss:.4f} motor_loss={motor_loss:.4f} "
                  f"cum_high_reward_choice={cum_high_reward_choice:.1f} trials={total_trials}")

        # Checkpoint
        if (upd + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_{upd+1:06d}.pt")
            save_checkpoint(ckpt_path, agent, optim_bandit, optim_motor, extra={"update": upd + 1})
            print(f"Saved checkpoint: {ckpt_path}")

        global_step += 1

    writer.close()
    ray.shutdown()


if __name__ == "__main__":
    main()