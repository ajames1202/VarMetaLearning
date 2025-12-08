# at top of bandit_train.py
import ray

from typing import Any, Tuple, Dict
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import argparse


import visual_bandit_env2 as vbe
import var_bandit_learner2 as bl
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import math
import pygame
import os
from datetime import datetime

from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter


import math, contextlib

# torch.backends.cudnn.enabled = True
# torch.autograd.set_detect_anomaly(True)

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


def save_checkpoint(path, agent, optim_bandit, optim_motor, extra):

    ckpt = {
        "model_state": agent.state_dict(),
        "optim_bandit_state": optim_bandit.state_dict(),
        "optim_motor_state": optim_motor.state_dict(),
        "extra": extra,
        "torch": torch.__version__,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

# def load_checkpoint(path, agent, optim_bandit=None, optim_motor=None, map_location="cpu"):
#     ckpt = torch.load(path, map_location=map_location)
#     agent.load_state_dict(ckpt["model_state"])
#     if optimizer is not None and "optim_state" in ckpt:
#         optimizer.load_state_dict(ckpt["optim_state"])
#     return ckpt.get("extra", {})




                    
def meta_ep_rollout(env, agent, device, session_K, session_N, worker_id=0, print_this_session=False):

    def log(*args, **kwargs):
        if print_this_session and worker_id == 0:
            print(*args, **kwargs)

    h_rnn = torch.zeros(1, agent.rnn.hidden_size, device=device)  # GRU hidden state
    attn_tokens = []  # [q0, o0, q1, o1, ...] each token is (1,H)

    obs, info = env.reset()
    done = False

    pair_index_counter = np.ones(session_K, dtype=np.int32) * -1
    high_reward_choice_per_N = np.zeros(session_N, dtype=np.int32)

    meta_ep_len = 0
    t = 0
    ep_start_flag = 1.0

    xy_pos_buf = []
    goal_vec_buf = []
    chosen_bandits_motor_buf = []

    left_obs = []
    right_obs = []

    chosen_bandits_buf = []
    bandit_rewards_buf = []
    meta_ep_start_buf = []

    choice_target = None
    p_left = torch.tensor(0.5)
    p_right = torch.tensor(0.5)

    while not done:
        obs_tensor = (
            torch.as_tensor(obs, device=device)
            .permute(2, 0, 1).unsqueeze(0)
            .to(torch.float32).div_(255.0)
        )

        # -----------------------------
        # CHOICE at trial start
        # -----------------------------
        if ep_start_flag == 1.0:
            left_view, right_view = extract_lr_views(obs_tensor, env, crop_size=112, pad=6)

            left_feats = agent.encode(left_view)    # (1,F)
            right_feats = agent.encode(right_view)  # (1,F)

            left_obs.append(left_view.squeeze(0).detach().cpu().numpy())
            right_obs.append(right_view.squeeze(0).detach().cpu().numpy())

            # Build query token (swap-invariant) and attend over history
            q_t = agent.pair_query_token(left_feats, right_feats)  # (1,H)
            attn_tokens.append(q_t)

            seq = torch.stack(attn_tokens, dim=0)   # (L,1,H)
            h_ctx = agent.attn(seq)[-1]             # (1,H) context at query position

            reward_logits = agent.reward_compute(h_ctx, left_feats, right_feats).squeeze(0)  # (2,)

            # Thompson sampling (same as your code)
            eps = 1e-4
            probs = torch.sigmoid(reward_logits).clamp(eps, 1.0 - eps)
            p_left, p_right = probs[0], probs[1]

            concentration = 5.0
            alpha_left  = p_left  * concentration + 1.0
            beta_left   = (1.0 - p_left)  * concentration + 1.0
            alpha_right = p_right * concentration + 1.0
            beta_right  = (1.0 - p_right) * concentration + 1.0

            dist_left  = torch.distributions.Beta(alpha_left,  beta_left)
            dist_right = torch.distributions.Beta(alpha_right, beta_right)

            u_left  = dist_left.rsample()
            u_right = dist_right.rsample()

            a_t = torch.argmax(torch.stack([u_left, u_right]))
            choice_target = F.one_hot(a_t, num_classes=2).unsqueeze(0).float()  # (1,2)

            meta_ep_len += 1
            ep_start_flag = 0.0

        # -----------------------------
        # MOTOR step
        # -----------------------------
        W, H = env.unwrapped.W, env.unwrapped.H
        x_pix, y_pix = env.unwrapped.cursor
        xy_norm = np.array([(x_pix/(W-1))*2-1, (y_pix/(H-1))*2-1], dtype=np.float32)

        left_c  = np.array([env.unwrapped.left_rect.centerx,  env.unwrapped.left_rect.centery],  np.float32)
        right_c = np.array([env.unwrapped.right_rect.centerx, env.unwrapped.right_rect.centery], np.float32)
        left_c_norm  = np.array([(left_c[0]/(W-1))*2-1,  (left_c[1]/(H-1))*2-1],  np.float32)
        right_c_norm = np.array([(right_c[0]/(W-1))*2-1, (right_c[1]/(H-1))*2-1], np.float32)

        chosen_center = left_c_norm if choice_target.argmax(dim=-1).item() == 0 else right_c_norm
        g_norm = chosen_center - xy_norm

        xy_pos_t   = torch.as_tensor(xy_norm).unsqueeze(0).to(device)
        goal_vec_t = torch.as_tensor(g_norm).unsqueeze(0).to(device)

        mu, log_std = agent.motor_fwd(choice_target.detach(), xy_pos=xy_pos_t, goal_vec=goal_vec_t)

        xy_pos_buf.append(xy_norm)
        goal_vec_buf.append(g_norm)
        chosen_bandits_motor_buf.append(choice_target.squeeze(0).detach().cpu().numpy())

        std = torch.exp(log_std)
        y = mu + std * torch.randn_like(std)
        action = torch.tanh(y)
        action_np = action.squeeze(0).detach().cpu().numpy()

        next_obs, reward, term, _, info = env.step(action_np)
        obs = next_obs
        done = term

        pair_idx_now = info.get("pair_index_in_session", -1)

        # -----------------------------
        # Trial ended => append outcome token + update GRU
        # -----------------------------
        if info.get("trial_ended"):
            meta_ep_start = 0.0 if meta_ep_len > 1 else 1.0
            meta_ep_start_torch = torch.tensor([[meta_ep_start]], device=device, dtype=torch.float32)

            a_oh = choice_target
            r_t  = torch.tensor([[float(reward)]], device=device, dtype=torch.float32)

            # rnn_fwd returns RAW GRU out for this trial (outcome token)
            o_t, h_rnn = agent.rnn_fwd(left_feats, right_feats, a_oh, r_t, meta_ep_start_torch, h_rnn)  # o_t: (1,H)
            attn_tokens.append(o_t)

            pair_index_ep = info.get("prev_pair_index_in_session", -1)
            pair_index_counter[pair_index_ep] += 1
            selected_high_reward = info.get("selected_high_reward_this_trial", False)
            flipped = info.get("side_is_flipped", False)

            if selected_high_reward:
                high_reward_choice_per_N[pair_index_counter[pair_index_ep]] += 1

            log(
                "Ep: ", info.get("trial_index"),
                ", idx = ", pair_idx_now,
                ", choice target:", choice_target.argmax(dim=-1).item(),
                ", Reached Target:", info.get("selected_target"),
                ", reward:", reward,
                ", flipped =", flipped,
                ", selected_high_reward = ", selected_high_reward,
                ", p_left =", round(float(p_left), 2),
                ", p_right =", round(float(p_right), 2)
            )

            chosen_bandits_buf.append(choice_target.squeeze(0).detach().cpu().numpy())
            bandit_rewards_buf.append(float(reward))
            meta_ep_start_buf.append(float(meta_ep_start))

            choice_target = None
            ep_start_flag = 1.0

        t += 1

    # session stats
    high_reward_choice_count_on_left = info.get("high_reward_choice_count_on_left", -1)
    high_reward_choice_count_on_right = info.get("high_reward_choice_count_on_right", -1)
    total_left_choices = info.get("total_left_choices", -1)
    total_right_choices = info.get("total_right_choices", -1)

    log(
        "Session finished. High-reward choices per N:", high_reward_choice_per_N,
        ", High-reward choices on left:", high_reward_choice_count_on_left,
        ", on right:", high_reward_choice_count_on_right,
        ", Total left choices:", total_left_choices,
        ", Total right choices:", total_right_choices,
    )

    high_reward_choice_per_N = high_reward_choice_per_N.astype(float)
    high_reward_choice_per_N /= float(session_K)
    high_reward_choice_per_N = high_reward_choice_per_N.round(2)

    cum_rewards = sum(bandit_rewards_buf)

    obs_bandit = {
        "left":  np.stack(left_obs,  axis=0),
        "right": np.stack(right_obs, axis=0),
    }

    return (
        xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
        obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf,
        high_reward_choice_per_N, cum_rewards
    )


@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    def __init__(self, session_K, session_N, seed=0, worker_id=0):
        self.session_K = session_K
        self.session_N = session_N
        self.device = torch.device("cpu")  # usually run envs on CPU
        self.worker_id = worker_id

        # each worker has its own env instance
        self.env = vbe.TwoChoiceReachingEnv(
            W=384,
            H=400,
            render_mode="rgb_array",
            seed=seed,
            session_K=session_K,
            session_N=session_N,
            trial_ms=3000,
            randomize_sides=True,
            shuffle=False,
        )

    def run_session(self, agent_state_dict, probs_this_session, print_this_session=False):
        # rebuild a fresh agent with same hyperparams as in main
        hidden_size = 128
        feature_dim = 128
        input_size = 2*feature_dim + 1 + 1

        agent = bl.BanditLearner(
            input_size=input_size,
            feature_dim=feature_dim,
            rnn_hidden_size=hidden_size,
            action_dim=2,
            num_pairs=self.session_K,
            max_trials=self.session_N * self.session_K,
        ).to(self.device)

        agent.load_state_dict(agent_state_dict)
        agent.eval()

        # set pair probabilities for this worker's env
        self.env.unwrapped.pair_probs = probs_this_session

        with torch.no_grad():
            return meta_ep_rollout(
                self.env, agent, self.device,
                self.session_K, self.session_N,
                worker_id=self.worker_id,
                print_this_session=print_this_session
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--episodes-per-update', type=int, default=8)  # <-- B
    parser.add_argument('--num-updates', type=int, default=500)        # <-- 500
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- TensorBoard writer ---
    log_dir = os.path.join("runs", f"bandit_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)


    session_K = 3
    session_N = 10

    hidden_size = 128
    feature_dim = 128
    input_size = 2*feature_dim + 1 + 1


    agent = bl.BanditLearner(
        input_size=input_size,
        feature_dim=feature_dim,
        rnn_hidden_size=hidden_size,
        action_dim=2,
        num_pairs=session_K,
        max_trials=session_N * session_K,
    )

    optim_bandit = torch.optim.Adam(agent.bandit_parameters(), lr=1e-3)
    optim_motor  = torch.optim.Adam(agent.motor_parameters(),  lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    # ---------- Ray init ----------
    # local: ray.init()
    # cluster: ray.init(address="auto")
    ray.init(ignore_reinit_error=True)

    # ---------- rollout workers ----------
    workers = [
        RolloutWorker.remote(session_K, session_N, seed=args.seed + 1000 * i, worker_id=i)
        for i in range(args.num_workers)
    ]


    num_updates = args.num_updates
    B = args.episodes_per_update
    W = args.num_workers

    for update_idx in range(num_updates):

        # --------------------------------------------------
        # Collect at least B sessions (episodes) for this update
        # --------------------------------------------------
        total_sessions_collected = 0

        # global buffers across all sessions in this update
        batch_xy_pos      = []
        batch_goal_vec    = []
        batch_chosen_bandits_motor = []
        batch_bandit_obs = []
        batch_chosen_bandits = []
        batch_bandit_rewards = []
        batch_meta_ep_start = []


        highR_perN_list   = []
        cum_rewards_list  = []

        while total_sessions_collected < B:

            # how many sessions still needed
            remaining = B - total_sessions_collected

            # use as many workers as needed, but not more than remaining
            num_launch = min(W, remaining)

            # ----- choose probs for each launched worker -----
            probs_list = []
            for w_id in range(num_launch):
                # (example: alternate high/low like before)
                # You can use any schedule you like here.
                global_ses = update_idx * B + total_sessions_collected + w_id
                # if global_ses % 2 == 0:
                #     probs_this_session = [(0.8, 0.2)] * session_K
                # else:
                #     probs_this_session = [(0.2, 0.8)] * session_K
                # L = np.random.uniform(0.1, 0.9, size=session_K)
                # L = [(0.8)]* session_K if np.random.rand() < 0.5 else [(0.2)]* session_K
                # R = 1.0 - L
                # L = [0.8] * session_K if np.random.rand() < 0.5 else [0.2] * session_K
                # R = [1.0 - x for x in L]

                probs_this_session = [(0.8, 0.2) if np.random.rand() < 0.5 else (0.2, 0.8) for _ in range(session_K)]
                probs_list.append(probs_this_session)

            # ---------- broadcast current weights (CPU) ----------
            agent_state_cpu = {k: v.detach().cpu() for k, v in agent.state_dict().items()}

            # ---------- launch rollouts in parallel ----------
            rollout_futures = []
            for w_id in range(num_launch):

                print_flag = (w_id == 0) and (total_sessions_collected == 0) and (update_idx%10 == 0)
                # meaning: only worker 0, only the first session of this update

                rollout_futures.append(
                    workers[w_id].run_session.remote(
                        agent_state_cpu,
                        probs_list[w_id],
                        print_this_session=print_flag
                    )
                )

            rollout_results = ray.get(rollout_futures)

            # ---------- aggregate results ----------
            for res in rollout_results:
                (xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf, 
                 obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf, high_reward_choice_per_N, cum_rewards) = res


                batch_xy_pos.extend(xy_pos_buf)
                batch_goal_vec.extend(goal_vec_buf)
                batch_chosen_bandits_motor.extend(chosen_bandits_motor_buf)
                batch_bandit_obs.append(obs_bandit)
                batch_chosen_bandits.append(torch.as_tensor(np.stack(chosen_bandits_buf), dtype=torch.float32))
                batch_bandit_rewards.append(torch.as_tensor(np.stack(bandit_rewards_buf), dtype=torch.float32))
                batch_meta_ep_start.append(torch.as_tensor(np.stack(meta_ep_start_buf), dtype=torch.float32))

                highR_perN_list.append(high_reward_choice_per_N)
                cum_rewards_list.append(cum_rewards)

                total_sessions_collected += 1
                if total_sessions_collected >= B:
                    break  # in case we overshoot slightly with num_launch


        # print(f"batch_rew len: {len(batch_xy_pos)}, batch_choice_tgts len: {len(batch_meta_ep_start)}")
        # --------------------------------------------------
        #   Now we have ~B sessions worth of data -> one update
        # --------------------------------------------------
        var_loss, motor_loss = agent.update2(
            optim_bandit, optim_motor,
            batch_xy_pos,
            batch_goal_vec,
            batch_chosen_bandits_motor,
            batch_bandit_obs,
            batch_chosen_bandits,
            batch_bandit_rewards,
            batch_meta_ep_start,
            device
        )

        # logging
        highR_perN_arr = np.stack(highR_perN_list)   # shape (B, N)
        mean_highR_perN = highR_perN_arr.mean(axis=0)
        mean_cum_rew = np.mean(cum_rewards_list)

        if update_idx % 10 == 0:
            print(f"[upd {update_idx:04d}] var_loss={var_loss:.4f} motor_loss={motor_loss:.4f} "
                  f"mean_cum_rew={mean_cum_rew:.1f}")


        # print(
        #     f"[Update {update_idx+1}/{num_updates}] "
        #     f"sessions_in_batch={total_sessions_collected}, "
        #     f"mean_cum_session_rewards={mean_cum_rew:.2f}, "
        #     f"ChoiceLoss={loss:.4f}, MotorLoss={policy_loss:.4f}, "
        #     f"mean_high_reward_choice_perN={mean_highR_perN}"
        # )

                # ---- TensorBoard logging ----
        global_step = update_idx  # one step per update

        # scalar losses
        writer.add_scalar("Loss/ChoiceLoss", var_loss, global_step)
        writer.add_scalar("Loss/MotorLoss", motor_loss, global_step)

        # rewards
        writer.add_scalar("Reward/MeanCumSession", mean_cum_rew, global_step)

        # per-N stats (either as histogram or individual scalars)
        # writer.add_histogram("Policy/HighRewardChoicePerN", mean_highR_perN, global_step)
        # or, if you want separate curves:
        for n, val in enumerate(mean_highR_perN):
            writer.add_scalar(f"Policy/HighRewardChoicePerN_N{n}", val, global_step)

        # if (update_idx + 1) % 10 == 0:
        #     for name, param in agent.named_parameters():
        #         writer.add_histogram(f"Params/{name}", param.detach().cpu().numpy(), global_step)



    ray.shutdown()
    writer.close()

            # if (ses + 1) % 10 == 0:  # every 10 sessions
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = getattr(args, "save_dir", "checkpoints")  # if you add CLI arg
    ckpt_path = os.path.join(save_dir, f"var_bandit_{stamp}.pt")
    extra = {
        "feature_dim": feature_dim,
        "hidden_size": hidden_size,
        "action_dim": 2,
    }
    save_checkpoint(ckpt_path, agent, optim_bandit, optim_motor, extra)
    print(f"Saved checkpoint to {ckpt_path}")

