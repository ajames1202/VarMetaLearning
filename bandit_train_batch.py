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

def extract_lr_views(obs_tensor, env, crop_size=112, pad=6):
    lr = env.unwrapped.left_rect
    rr = env.unwrapped.right_rect
    _, _, H, W = obs_tensor.shape

    y1 = max(min(lr.top, rr.top) - pad, 0)
    y2 = min(max(lr.bottom, rr.bottom) + pad, H)

    def crop_resize(rect):
        x1 = max(rect.left - pad, 0)
        x2 = min(rect.right + pad, W)
        patch = obs_tensor[:, :, y1:y2, x1:x2]
        patch = F.interpolate(patch, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
        return patch

    return crop_resize(lr), crop_resize(rr)


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


def run_pair_idx_probe(obs_bandit_batch, pair_idx_batch, device="cuda",
                       steps=400, lr=2e-2, wd=1e-4, test_frac=0.2, batch=2048):
    import numpy as np
    import torch
    import torch.nn as nn

    X = torch.as_tensor(np.concatenate(obs_bandit_batch, axis=0), device=device, dtype=torch.float32)  # (N,F)
    y = torch.as_tensor(np.concatenate(pair_idx_batch, axis=0), device=device, dtype=torch.long)      # (N,)

    # drop unknown labels if any
    m = y >= 0
    X, y = X[m], y[m]
    if X.numel() == 0:
        print("[pair_idx probe] no data")
        return None

    K = int(y.max().item()) + 1
    N = X.size(0)

    # split
    perm = torch.randperm(N, device=device)
    n_test = max(1, int(N * test_frac))
    te = perm[:n_test]
    tr = perm[n_test:]

    # standardize for stability
    mean = X[tr].mean(0, keepdim=True)
    std  = X[tr].std(0, keepdim=True).clamp_min(1e-6)
    Xs = (X - mean) / std

    probe = nn.Linear(Xs.size(1), K).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(steps):
        idx = tr[torch.randint(0, tr.numel(), (min(batch, tr.numel()),), device=device)]
        loss = loss_fn(probe(Xs[idx]), y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc_tr = (probe(Xs[tr]).argmax(-1) == y[tr]).float().mean().item()
        acc_te = (probe(Xs[te]).argmax(-1) == y[te]).float().mean().item()

    print(f"[pair_idx probe] K={K} N={N} train_acc={acc_tr:.3f} test_acc={acc_te:.3f} chance={1.0/K:.3f}")
    return acc_te


# -------------------------
# Rollout with Query + Memory tokens
# -------------------------
@torch.no_grad()
def meta_ep_rollout(env, agent, device, session_K: int, session_N: int,
                   worker_id: int = 0, print_this_session: bool = False):

    def log(*args, **kwargs):
        if print_this_session and worker_id == 0:
            print(*args, **kwargs)

    agent.eval()

    # Token history: rows are tokens of size agent.input_size = F + 2 + 1 + 1
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

    left_c  = np.array([env.unwrapped.left_rect.centerx,  env.unwrapped.left_rect.centery],  np.float32)
    right_c = np.array([env.unwrapped.right_rect.centerx, env.unwrapped.right_rect.centery], np.float32)
    left_c_norm  = np.array([left_c[0]  * sx + ox, left_c[1]  * sy + oy], np.float32)
    right_c_norm = np.array([right_c[0] * sx + ox, right_c[1] * sy + oy], np.float32)

    # ---------- reusable tensors ----------
    xy_pos_t   = torch.empty((1, 2), device=device, dtype=torch.float32)
    goal_vec_t = torch.empty((1, 2), device=device, dtype=torch.float32)

    # Buffers (motor)
    xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf = [], [], []

    # Buffers (bandit) — NOTE: now we store featsL and featsR separately
    featsL_buf, featsR_buf = [], []
    chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf = [], [], []
    pair_idx_buf = []

    meta_trial_idx = 0
    ep_start_flag = 1.0

    # cached per-trial values (for motor steps + trial end)
    trial_fL = None                 # torch (F,)
    trial_fR = None                 # torch (F,)
    trial_fL_np = None              # np (F,)
    trial_fR_np = None              # np (F,)
    trial_action_1d = None          # torch (2,)
    trial_action_2d = None          # torch (1,2)
    trial_action_idx = 0
    trial_action_np = None          # np (2,)
    trial_meta_start = 0.0

    while not done:

        # -----------------------
        # TRIAL START: encode L/R + append QL,QR + choose arm
        # -----------------------
        if ep_start_flag == 1.0:
            trial_meta_start = 1.0 if meta_trial_idx == 0 else 0.0

            # obs -> torch
            obs_tensor = torch.from_numpy(obs).to(device).permute(2, 0, 1).unsqueeze(0).float()
            obs_tensor.mul_(1.0 / 255.0)

            # Encode LEFT and RIGHT separately
            left_view, right_view = extract_lr_views(obs_tensor, env, crop_size=112, pad=6)
            fL = agent.encode(left_view).squeeze(0)   # (F,)
            fR = agent.encode(right_view).squeeze(0)  # (F,)

            trial_fL, trial_fR = fL, fR
            trial_fL_np = fL.detach().cpu().numpy()
            trial_fR_np = fR.detach().cpu().numpy()

            # Build QL and QR tokens: [feat, zeros_action(2), zeros_reward(1), meta_start(1)]
            zeros_a = torch.zeros((2,), device=device, dtype=torch.float32)
            zeros_r = torch.zeros((1,), device=device, dtype=torch.float32)
            mstart  = torch.tensor([trial_meta_start], device=device, dtype=torch.float32)

            xQL = torch.cat([fL, zeros_a, zeros_r, mstart], dim=0)  # (D,)
            xQR = torch.cat([fR, zeros_a, zeros_r, mstart], dim=0)  # (D,)

            history = torch.cat([history, xQL.unsqueeze(0), xQR.unsqueeze(0)], dim=0)  # add 2 tokens

            # Causal transformer over history; take last two states as the query states
            out = agent.transformer_fwd_tokens(history)  # (S,H)
            hQL, hQR = out[-2], out[-1]                  # (H,), (H,)

            # Per-arm logits (Bernoulli)
            left_logit  = agent.reward_compute(hQL, fL)  # scalar
            right_logit = agent.reward_compute(hQR, fR)  # scalar
            probs = torch.sigmoid(torch.stack([left_logit, right_logit], dim=0)).clamp(1e-4, 1.0 - 1e-4)

            # Thompson sampling (same as your previous rollout)
            concentration = 5.0
            alpha = probs * concentration + 1.0
            beta  = (1.0 - probs) * concentration + 1.0
            u = torch.stack([
                torch.distributions.Beta(alpha[0], beta[0]).rsample(),
                torch.distributions.Beta(alpha[1], beta[1]).rsample()
            ])
            a_t = torch.argmax(u)

            trial_action_idx = int(a_t.item())
            trial_action_1d = F.one_hot(a_t, num_classes=2).float()          # (2,)
            trial_action_2d = trial_action_1d.unsqueeze(0)                   # (1,2)
            trial_action_np = trial_action_1d.detach().cpu().numpy()         # (2,)

        # -----------------------
        # MOTOR step
        # -----------------------
        x_pix, y_pix = env.unwrapped.cursor
        x_norm = x_pix * sx + ox
        y_norm = y_pix * sy + oy

        goal_center = left_c_norm if trial_action_idx == 0 else right_c_norm
        g0 = goal_center[0] - x_norm
        g1 = goal_center[1] - y_norm

        xy_pos_t[0, 0] = float(x_norm)
        xy_pos_t[0, 1] = float(y_norm)
        goal_vec_t[0, 0] = float(g0)
        goal_vec_t[0, 1] = float(g1)

        mu, log_std = agent.motor_fwd(trial_action_2d, xy_pos=xy_pos_t, goal_vec=goal_vec_t)
        std = torch.exp(log_std)
        y = mu + std * torch.randn_like(std)
        action = torch.tanh(y)
        action_np = action.squeeze(0).cpu().numpy()

        next_obs, reward, term, trunc, info = env.step(action_np)
        done = bool(term) or bool(trunc)

        # save motor buffers
        xy_pos_buf.append(np.array([x_norm, y_norm], np.float32))
        goal_vec_buf.append(np.array([g0, g1], np.float32))
        chosen_bandits_motor_buf.append(trial_action_np)

        # -----------------------
        # TRIAL END: append M token + store bandit buffers
        # -----------------------
        if info.get("trial_ended", False):
            # Memory token uses CHOSEN feature + (action, reward, meta_start)
            feat_chosen = trial_fL if trial_action_idx == 0 else trial_fR  # (F,)
            r = torch.tensor([float(reward)], device=device, dtype=torch.float32)
            mstart = torch.tensor([trial_meta_start], device=device, dtype=torch.float32)

            xM = torch.cat([feat_chosen, trial_action_1d, r, mstart], dim=0)  # (D,)
            history = torch.cat([history, xM.unsqueeze(0)], dim=0)

            # store bandit buffers per TRIAL
            featsL_buf.append(trial_fL_np)
            featsR_buf.append(trial_fR_np)
            chosen_bandits_buf.append(trial_action_np)
            bandit_rewards_buf.append(float(reward))
            meta_ep_start_buf.append(float(trial_meta_start))

            selected_high_reward = info.get("selected_high_reward_this_trial", -1)
            if selected_high_reward:
                idxN = min(int(info.get("trial_index_in_pair", 0)), session_N - 1)
                high_reward_choice_per_N[idxN] += 1.0
                cum_high_reward_choice += 1.0

            reached_target = info.get("selected_target")
            curr_pair_index = info.get("prev_pair_index_in_session", -1)
            pair_idx_buf.append(int(curr_pair_index))

            flipped = info.get("side_is_flipped", -1)

            log(
                f"(RolloutWorker pid={os.getpid()}) [W{worker_id}] trial={meta_trial_idx} "
                f"reward={float(reward):.1f}, pair_idx={curr_pair_index}, choice={trial_action_idx}, "
                f"reached_target={reached_target}, flipped={flipped}, selected_high_reward={selected_high_reward}"
            )

            meta_trial_idx += 1
            ep_start_flag = 1.0
        else:
            ep_start_flag = 0.0

        obs = next_obs

    # stack bandit buffers (T,F), (T,2), (T,)
    featsL_arr = np.stack(featsL_buf, axis=0) if len(featsL_buf) else np.zeros((0, 0), np.float32)
    featsR_arr = np.stack(featsR_buf, axis=0) if len(featsR_buf) else np.zeros((0, 0), np.float32)
    chosen_arr = np.stack(chosen_bandits_buf, axis=0) if len(chosen_bandits_buf) else np.zeros((0, 2), np.float32)
    rewards_arr = np.array(bandit_rewards_buf, np.float32)
    start_arr = np.array(meta_ep_start_buf, np.float32)

    return {
        # motor
        "xy_pos_buf": xy_pos_buf,
        "goal_vec_buf": goal_vec_buf,
        "chosen_bandits_motor_buf": chosen_bandits_motor_buf,

        # bandit (NEW keys for 3-token update2)
        "featsL": featsL_arr,                  # (T,F)
        "featsR": featsR_arr,                  # (T,F)
        "chosen_bandits_buf": chosen_arr,      # (T,2)
        "bandit_rewards_buf": rewards_arr,     # (T,)
        "meta_ep_start_buf": start_arr,        # (T,)
        "pair_idx_buf": pair_idx_buf,

        "metrics": {
            "cum_high_reward_choice": float(cum_high_reward_choice),
            "high_reward_choice_per_N": high_reward_choice_per_N,
            "num_trials": int(len(bandit_rewards_buf)),
        }
    }

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

        feature_dim = 192
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
    parser.add_argument("--feature-dim", type=int, default=192)
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
        featsL_batch = []
        featsR_batch = []
        chosen_bandits_batch = []      # list of (T,2)
        bandit_rewards_batch = []      # list of (T,)
        meta_ep_start_batch = []       # list of (T,)
        pair_idx_batch = []   # list of (T,)


        cum_high_reward_choice = 0.0
        total_trials = 0



        for r in results:
            # motor buffers (flat)
            xy_pos_buf += r["xy_pos_buf"]
            goal_vec_buf += r["goal_vec_buf"]
            chosen_bandits_motor_buf += r["chosen_bandits_motor_buf"]

            # bandit buffers (per episode / per worker)
            featsL_batch.append(r["featsL"])                 # (T,F)
            featsR_batch.append(r["featsR"])                 # (T,F)
            chosen_bandits_batch.append(r["chosen_bandits_buf"])   # (T,2)
            bandit_rewards_batch.append(r["bandit_rewards_buf"])   # (T,)
            meta_ep_start_batch.append(r["meta_ep_start_buf"])     # (T,)

            # optional debug/probes
            pair_idx_batch.append(np.array(r.get("pair_idx_buf", []), np.int64))

            cum_high_reward_choice += r["metrics"]["cum_high_reward_choice"]
            total_trials += r["metrics"]["num_trials"]

        cum_high_reward_choice /= args.episodes_per_update

        # # OPTIONAL safety: ensure all episodes same T (update2 assumes this)
        # T0 = obs_bandit_batch[0].shape[0]
        # obs_bandit_batch = [x for x in obs_bandit_batch if x.shape[0] == T0]
        # chosen_bandits_batch = chosen_bandits_batch[:len(obs_bandit_batch)]
        # bandit_rewards_batch = bandit_rewards_batch[:len(obs_bandit_batch)]
        # meta_ep_start_batch = meta_ep_start_batch[:len(obs_bandit_batch)]

        # Train update
        agent.train()
        var_loss, motor_loss = agent.update2(
            optim_bandit=optim_bandit,
            optim_motor=optim_motor,
            xy_pos_buf=xy_pos_buf,
            goal_vec_buf=goal_vec_buf,
            chosen_bandits_motor_buf=chosen_bandits_motor_buf,
            featsL=featsL_batch,
            featsR=featsR_batch,
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

        # if upd == 10:
        #     run_pair_idx_probe(obs_bandit_batch, pair_idx_batch, device=train_device)


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