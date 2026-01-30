# at top of bandit_train.py
import ray
# ray.init(include_dashboard=False)

from PIL import Image
from typing import Any, Tuple, Dict
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import argparse


import visual_bandit_env3 as vbe
import var_bandit_learner3 as bl
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import math
import pygame
import os
from datetime import datetime

from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter


import math, contextlib
import matplotlib.pyplot as plt


# torch.backends.cudnn.enabled = True
# torch.autograd.set_detect_anomaly(True)


TOP_GROUPS = [
    "enc", "attn", "q_in", "ctx_to_logit",
    "mlp_pos", "mlp_goal", "mu_head", "log_std_head",
]

def log_group_weight_updates(writer, agent, before, step, tag="Upd"):
    """
    before: dict(name -> tensor clone) from agent.named_parameters()
    Logs ~8-16 curves total: L2 and max|Δ| per top-level module group.
    """
    # accumulators
    sq = {g: 0.0 for g in TOP_GROUPS}
    mx = {g: 0.0 for g in TOP_GROUPS}

    with torch.no_grad():
        for name, p in agent.named_parameters():
            if name not in before:
                continue
            g = name.split(".", 1)[0]  # top-level: enc.*, attn.*, ...
            if g not in sq:
                continue
            d = (p - before[name])
            sq[g] += float(d.pow(2).sum().item())
            mx[g] = max(mx[g], float(d.abs().max().item()))

    for g in TOP_GROUPS:
        writer.add_scalar(f"{tag}/L2/{g}", math.sqrt(sq[g]), step)
        writer.add_scalar(f"{tag}/MaxAbs/{g}", mx[g], step)


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

                    
def meta_ep_rollout(env, agent, device, session_K, session_N, worker_id=0, print_this_session=False):
    """
    Rollout consistent with update2() in var_bandit_learner2_3mha_complete.py:
      - action selection uses policy_net_il / policy_net_obs
      - teacher/self latent state variables updated via 3 encoder MHAs with key_padding_mask (no manual filtering)
      - preserves env protocol:
          * episode end: term from env.step
          * trial end: info.get("trial_ended")
          * teacher action on AO-o/OL-o: info.get("selected_target")
      - AO-o reward is stored as 2 (NO_FEEDBACK) to match training convention
    """

    def log(*args, **kwargs):
        if print_this_session and worker_id == 0:
            print(*args, **kwargs)

    # -----------------------------
    # Helpers: compute last-step z using masked MHA over full history tokens
    # -----------------------------
    @torch.no_grad()
    def _infer_last_z(x_seq, keep_mask_tb, encoder, ln):
        L = x_seq.size(0)
        if L == 0:
            return torch.zeros(agent.z_dim, device=device)

        # key_padding_mask wants (B,T) True=ignore. Here B=1.
        kpm = (~keep_mask_tb).view(1, L)

        # causal mask (same shape as x_seq length)
        attn_mask = torch.triu(
            torch.ones((L, L), device=device, dtype=torch.bool),
            diagonal=1
        )

        # IMPORTANT: match update2(): BOS-safe attention + residual + LN
        h = agent.safe_attn(encoder, x_seq, attn_mask, kpm)   # (L,1,H), BOS handled inside
        h = ln(h + x_seq)

        q = agent.post_head(h[-1])                            # (1,2*z)
        mu = q[..., :agent.z_dim].squeeze(0)                  # (z,)
        return mu
    
    @torch.no_grad()
    def fuse_belief(mu_self_q: torch.Tensor, which: str) -> torch.Tensor:
        """
        mu_self_q: (z_dim,)
        which: "AO" or "OL"
        Returns fused mean mu_fused_* (z_dim,)
        """
        L = len(z_ac_tokens)
        if L == 0:
            return mu_self_q

        # keys/values are all past z_ac means: (L,1,z)
        kv = torch.stack(z_ac_tokens, dim=0).unsqueeze(1)

        if which == "AO":
            keep = torch.as_tensor([c in ("AO-o", "AO-s") for c in cond_ids],
                                device=device, dtype=torch.bool)
        else:
            keep = torch.as_tensor([c in ("OL-o", "OL-s") for c in cond_ids],
                                device=device, dtype=torch.bool)

        # If no valid entries, skip fusion (prevents all-masked attention NaNs)
        if keep.sum() == 0:
            return mu_self_q

        # key_padding_mask: (B,L) True=ignore
        kpm = (~keep).view(1, L)

        q = mu_self_q.view(1, 1, -1)             # (1,1,z)
        ctx, _ = agent.fuse_attn(q, kv, kv, key_padding_mask=kpm)  # (1,1,z) :contentReference[oaicite:4]{index=4}
        fused = agent.fuse_post_head(torch.cat([q, ctx], dim=-1))   # (1,1,2z) :contentReference[oaicite:5]{index=5}
        mu_fused = fused[..., :agent.z_dim].view(-1)               # (z,)
        return mu_fused




    # -----------------------------
    # Memory: store full event tokens so encoder inference matches training x_all
    # -----------------------------
    x_tokens = []  # list of (1,H) event tokens in time order
    actor_ids = [] # list of int: 0=self,1=teacher
    cond_ids = []  # list of str: "IL","AO-s","AO-o","OL-s","OL-o"
    z_ac_tokens = []  # list of (z_dim,) posterior means aligned with x_tokens
    mu_self_last = torch.zeros(agent.z_dim, device=device)  # last self-stream belief (mean)


    # -----------------------------
    # Latent state vars used by update2 policy loop
    # -----------------------------
    z_il_tm1   = torch.zeros(agent.z_dim, device=device)
    z_ao_s_tm1 = torch.zeros(agent.z_dim, device=device)
    z_ol_s_tm1 = torch.zeros(agent.z_dim, device=device)
    z_ao_o_last = torch.zeros(agent.z_dim, device=device)
    z_ol_o_last = torch.zeros(agent.z_dim, device=device)

    obs, info = env.reset()
    done = False

    # bookkeeping arrays expected downstream (kept consistent with your current code)
    pair_index_counter_il = np.ones(session_K, dtype=np.int32) * -1
    pair_index_counter_ao = np.ones(session_K, dtype=np.int32) * -1
    pair_index_counter_ol = np.ones(session_K, dtype=np.int32) * -1
    high_reward_choices_il = np.zeros(session_N, dtype=np.int32)
    high_reward_choice_ao = np.zeros(session_N, dtype=np.int32)
    high_reward_choice_ol = np.zeros(session_N, dtype=np.int32)

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
    trial_cond_buf = []
    actor_buf = []

    # per-trial cached
    choice_target = None
    curr_trial_condition = None
    left_feats = None
    right_feats = None
    inp_pair = None
    x_dec = None
    a_t = None
    probs = None

    while not done:
        obs_tensor = (
            torch.as_tensor(obs, device=device)
            .permute(2, 0, 1).unsqueeze(0)
            .to(torch.float32).div_(255.0)
        )

        # -----------------------------
        # Trial start: observe stimuli + choose action using policy_net
        # -----------------------------
        if ep_start_flag == 1.0:
            left_view, right_view = extract_lr_views(obs_tensor, env, crop_size=112, pad=6)
            curr_trial_condition = info.get("curr_trial_condition")

            left_feats = agent.encode(left_view)    # (1,F)
            right_feats = agent.encode(right_view)  # (1,F)

            left_obs.append(left_view.squeeze(0).detach().cpu().numpy())
            right_obs.append(right_view.squeeze(0).detach().cpu().numpy())

            lr_concat = torch.cat([left_feats, right_feats], dim=-1)  # (1,2F)
            inp_pair = agent.inp_emb(lr_concat)                       # (1,H)

            # training uses x_dec = inp + actor_emb
            actor_self = agent.actor(torch.tensor([0], device=device, dtype=torch.long))  # (1,H)
            x_dec = inp_pair + actor_self  # (1,H)
            tau = 0.2  # keep your sampling temp

            if curr_trial_condition in ("IL", "AO-s", "OL-s"):

                if curr_trial_condition == "IL":
                    belief = mu_self_last
                elif curr_trial_condition == "AO-s":
                    belief = fuse_belief(mu_self_last, which="AO")
                else:  # "OL-s"
                    belief = fuse_belief(mu_self_last, which="OL")

                q_logits = agent.q_net(torch.cat([x_dec.squeeze(0), belief], dim=-1)).view(1, 2)  # :contentReference[oaicite:8]{index=8}
                q_logits = q_logits / tau

                dist = torch.distributions.Categorical(logits=q_logits)
                probs = dist.probs.detach().cpu().numpy()
                a_t = dist.sample().squeeze().to(torch.long)
                choice_target = F.one_hot(a_t, num_classes=2).unsqueeze(0).float()

            else:
                # AO-o / OL-o : teacher acts
                a_env = info.get("selected_target", None)
                if a_env is not None and int(a_env) in (0, 1):
                    a_t = torch.tensor(int(a_env), device=device, dtype=torch.long)
                else:
                    a_t = torch.tensor(0, device=device, dtype=torch.long)
                choice_target = F.one_hot(a_t, num_classes=2).unsqueeze(0).float()


            meta_ep_len += 1
            ep_start_flag = 0.0

        # -----------------------------
        # Motor step (exact env usage)
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

        xy_pos_t = torch.as_tensor(xy_norm, device=device).unsqueeze(0)
        goal_vec_t = torch.as_tensor(g_norm, device=device).unsqueeze(0)

        # print("choice_target.shape=", choice_target.shape, ", xy_pos_t.shape=", xy_pos_t.shape, ", goal_vec_t.shape=", goal_vec_t.shape)
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

        # -----------------------------
        # Trial end (exact key: trial_ended)
        # -----------------------------
        if info.get("trial_ended"):
            meta_ep_start = 0.0 if meta_ep_len > 1 else 1.0

            actor_id = 0
            # On observation trials, set action from env
            if (curr_trial_condition == "AO-o") or (curr_trial_condition == "OL-o"):
                a_env = info.get("selected_target", -1)
                a_t = torch.tensor(int(a_env), device=device, dtype=torch.long)
                choice_target = F.one_hot(a_t, num_classes=2).unsqueeze(0).float()
                actor_id = 1

            # Reward encoding: AO-o has NO_FEEDBACK=2
            r_idx = int(reward)
            if curr_trial_condition == "AO-o":
                r_idx = 2

            # Build event token x_all = inp + action + actor + reward  (matches training x_all)
            action_emb = agent.action(a_t.view(1))  # (1,H)
            actor_emb  = agent.actor(torch.tensor([actor_id], device=device, dtype=torch.long))  # (1,H)
            reward_emb = agent.rwd_in(torch.tensor([r_idx], device=device, dtype=torch.long))    # (1,H)

            x_event = inp_pair + action_emb + actor_emb + reward_emb  # (1,H)

            x_tokens.append(x_event)
            actor_ids.append(actor_id)
            cond_ids.append(curr_trial_condition)

            # -----------------------------
            # Latent updates via 3 MHAs with masking (no manual filtering)
            # -----------------------------
            x_seq = torch.stack(x_tokens, dim=0)  # (L,1,H)
            L = x_seq.size(0)

            actor_t = torch.as_tensor(actor_ids, device=device, dtype=torch.long)  # (L,)
            cond_list = cond_ids  # python list

            # keep masks for keys/values
            keep_self = (actor_t == 0)  # self events
            keep_ao_teacher = torch.as_tensor([c == "AO-o" for c in cond_list], device=device, dtype=torch.bool)
            keep_ol_teacher = torch.as_tensor([c == "OL-o" for c in cond_list], device=device, dtype=torch.bool)

            # infer last-step posterior mean z for relevant stream
            if curr_trial_condition in ("IL", "AO-s", "OL-s"):
                z_last_self = _infer_last_z(x_seq, keep_self, agent.enc_self, agent.self_ln)
                if curr_trial_condition == "IL":
                    z_il_tm1 = z_last_self
                elif curr_trial_condition == "AO-s":
                    z_ao_s_tm1 = z_last_self
                else:
                    z_ol_s_tm1 = z_last_self

            elif curr_trial_condition == "AO-o":
                z_ao_o_last = _infer_last_z(x_seq, keep_ao_teacher, agent.enc_teacher_ao, agent.teacher_ao_ln)
                # z_ao_o_last = z_last_ao

            elif curr_trial_condition == "OL-o":
                z_ol_o_last = _infer_last_z(x_seq, keep_ol_teacher, agent.enc_teacher_ol, agent.teacher_ol_ln)
                # z_ol_o_last = z_last_ol

            
            if curr_trial_condition in ("IL", "AO-s", "OL-s"):
                # your _infer_last_z(...) for keep_self should have produced the mean for this just-finished self event
                mu_event = _infer_last_z(x_seq, keep_self, agent.enc_self, agent.self_ln)
                mu_self_last = mu_event
            elif curr_trial_condition == "AO-o":
                mu_event = z_ao_o_last
            else:  # "OL-o"
                mu_event = z_ol_o_last

            z_ac_tokens.append(mu_event.detach())
    

            # -----------------------------
            # Bookkeeping (unchanged)
            # -----------------------------
            pair_index_ep = info.get("prev_pair_index_in_session", -1)
            selected_high_reward = info.get("selected_high_reward_this_trial", -1)
            flipped = info.get("prev_side_is_flipped", False)

            if curr_trial_condition == "IL":
                pair_index_counter_il[pair_index_ep] += 1
            elif curr_trial_condition == "AO-s":
                pair_index_counter_ao[pair_index_ep] += 1
            elif curr_trial_condition == "OL-s":
                pair_index_counter_ol[pair_index_ep] += 1

            if (curr_trial_condition == "IL") and selected_high_reward:
                high_reward_choices_il[pair_index_counter_il[pair_index_ep]] += 1
            elif (curr_trial_condition == "AO-s") and selected_high_reward:
                high_reward_choice_ao[pair_index_counter_ao[pair_index_ep]] += 1
            elif (curr_trial_condition == "OL-s") and selected_high_reward:
                high_reward_choice_ol[pair_index_counter_ol[pair_index_ep]] += 1

            log(
                "Ep: ", info.get("trial_index"),
                ", curr_idx = ", pair_index_ep,
                ", iter_il = ", pair_index_counter_il[pair_index_ep],
                ", iter_ao = ", pair_index_counter_ao[pair_index_ep],
                ", iter_ol = ", pair_index_counter_ol[pair_index_ep],
                ", curr_trial_condition =", curr_trial_condition,
                ", choice target:", int(choice_target.argmax(dim=-1).item()),
                ", Reached Target:", info.get("selected_target"),
                ", reward:", r_idx,
                ", flipped =", flipped,
                ", selected_high_reward = ", selected_high_reward,
                ", probs = ", probs,
                ", alpha =", env.unwrapped.alpha,
                ", tau =", env.unwrapped.tau
            )

            chosen_bandits_buf.append(choice_target.squeeze(0).detach().cpu().numpy())
            bandit_rewards_buf.append(float(r_idx))
            meta_ep_start_buf.append(float(meta_ep_start))
            actor_buf.append(int(actor_id))
            trial_cond_buf.append(curr_trial_condition)

            # reset for next trial
            choice_target = None
            ep_start_flag = 1.0

        t += 1

    # Summary returns (keep existing structure)
    bandit_rewards_arr = np.asarray(bandit_rewards_buf, dtype=np.float32)
    trial_cond_arr = np.asarray(trial_cond_buf)

    cum_rewards_il = float(bandit_rewards_arr[trial_cond_arr == "IL"].sum())
    cum_rewards_ao = float(bandit_rewards_arr[trial_cond_arr == "AO-s"].sum())
    cum_rewards_ol = float(bandit_rewards_arr[trial_cond_arr == "OL-s"].sum())

    high_reward_choices_il = np.array(high_reward_choices_il, dtype=np.int32)
    high_reward_choices_ao = np.array(high_reward_choice_ao, dtype=np.int32)
    high_reward_choices_ol = np.array(high_reward_choice_ol, dtype=np.int32)

    obs_bandit = {
        "left":  np.stack(left_obs, axis=0),
        "right": np.stack(right_obs, axis=0),
    }

    return (
        xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
        obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf,
        actor_buf, trial_cond_buf,
        cum_rewards_il, cum_rewards_ao, cum_rewards_ol,
        high_reward_choices_il, high_reward_choices_ao, high_reward_choices_ol
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
            randomize_sides=False,
            shuffle=True,
        )

    def run_session(self, agent_state_dict, probs_this_session, print_this_session=False):
        # rebuild a fresh agent with same hyperparams as in main
        hidden_size = 128
        feature_dim = 128
        input_size = feature_dim + 1

        agent = bl.BanditLearner(
            input_size=input_size,
            feature_dim=feature_dim,
            rnn_hidden_size=hidden_size,
            action_dim=2
        ).to(self.device)

        agent.load_state_dict(agent_state_dict)
        agent.eval()

        # set pair probabilities for this worker's env
        self.env.unwrapped.pair_probs = probs_this_session
        self.env.unwrapped.alpha = [0, np.random.choice([0.09, 0.2]), np.random.choice([0.09, 0.2])]
        self.env.unwrapped.tau = [0.288, 0.288, 0.288]

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
    parser.add_argument('--num-updates', type=int, default=1000)        # <-- 500
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # --- TensorBoard writer ---
    log_dir = os.path.join("runs", f"bandit_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)


    session_K = 3
    session_N = 12

    hidden_size = 128
    feature_dim = 128
    input_size = feature_dim + 1

    ema = None
    ema_beta = 0.9
    best_ema = -float("inf")
    patience = 100
    min_delta = 0.1
    bad = 0
    warmup_updates = 400



    agent = bl.BanditLearner(
        input_size=input_size,
        feature_dim=feature_dim,
        rnn_hidden_size=hidden_size,
        action_dim=2
    )

    optim_bandit = torch.optim.Adam(agent.bandit_parameters(), lr=1e-4)
    optim_motor  = torch.optim.Adam(agent.motor_parameters(),  lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    # ---------- Ray init ----------
    # local: ray.init()
    # cluster: ray.init(address="auto")
    # ray.init(ignore_reinit_error=True)
    ray.init(include_dashboard=False)

    # ---------- rollout workers ----------
    workers = [
        RolloutWorker.remote(session_K, session_N, seed=args.seed + 1000 * i, worker_id=i)
        for i in range(args.num_workers)
    ]

    # ---------- Collect teacher data ----------

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
        batch_actor = []
        batch_trial_cond = []


        highR_perN_list   = []
        cum_rewards_list_il  = []
        cum_rewards_list_ao  = []
        cum_rewards_list_ol  = []

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
                # probs_this_session = [(0.8, 0.2)] * session_K ## Sanity check
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
                 obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf, 
                actor_buf, trial_cond_buf,
                 cum_rewards_il, cum_rewards_ao, cum_rewards_ol, high_reward_choices_il, high_reward_choices_ao, high_reward_choices_ol) = res


                batch_xy_pos.extend(xy_pos_buf)
                batch_goal_vec.extend(goal_vec_buf)
                batch_chosen_bandits_motor.extend(chosen_bandits_motor_buf)
                batch_bandit_obs.append(obs_bandit)
                batch_chosen_bandits.append(torch.as_tensor(np.stack(chosen_bandits_buf), dtype=torch.float32))
                batch_bandit_rewards.append(torch.as_tensor(np.stack(bandit_rewards_buf), dtype=torch.float32))
                batch_meta_ep_start.append(torch.as_tensor(np.stack(meta_ep_start_buf), dtype=torch.float32))
                batch_actor.append(torch.as_tensor(np.stack(actor_buf), dtype=torch.long))
                batch_trial_cond.append(np.array(trial_cond_buf))



                cum_rewards_list_il.append(cum_rewards_il)
                cum_rewards_list_ao.append(cum_rewards_ao)
                cum_rewards_list_ol.append(cum_rewards_ol)

                total_sessions_collected += 1
                if total_sessions_collected >= B:
                    break  # in case we overshoot slightly with num_launch


        # print(f"batch_rew len: {len(batch_xy_pos)}, batch_choice_tgts len: {len(batch_meta_ep_start)}")
        # --------------------------------------------------
        #   Now we have ~B sessions worth of data -> one update
        # --------------------------------------------------
        
        # --- snapshot weights before update ---
        # before = {n: p.detach().clone() for n, p in agent.named_parameters()}


        var_loss, motor_loss = agent.update2(
            optim_bandit, optim_motor,
            batch_xy_pos,
            batch_goal_vec,
            batch_chosen_bandits_motor,
            batch_bandit_obs,
            batch_chosen_bandits,
            batch_bandit_rewards,
            batch_meta_ep_start,
            batch_actor,
            batch_trial_cond,
            device
        )

        # --- log grouped parameter updates (Δw) ---
        # log_group_weight_updates(writer, agent, before, update_idx, tag="Upd")



        # logging
        # highR_perN_arr = np.stack(highR_perN_list)   # shape (B, N)
        # mean_highR_perN = highR_perN_arr.mean(axis=0)
        mean_cum_rew_il = np.mean(cum_rewards_list_il)
        mean_cum_rew_ao = np.mean(cum_rewards_list_ao)
        mean_cum_rew_ol = np.mean(cum_rewards_list_ol)

        train_score = mean_cum_rew_il + mean_cum_rew_ao + mean_cum_rew_ol

        ema = train_score if ema is None else (ema_beta * ema + (1 - ema_beta) * train_score)



        if update_idx % 10 == 0:
            print(f"[upd {update_idx:04d}] var_loss={var_loss:.4f} motor_loss={motor_loss:.4f} "
                  f"mean_cum_rew_il={mean_cum_rew_il:.1f} "
                  f"mean_cum_rew_ao={mean_cum_rew_ao:.1f} "
                  f"mean_cum_rew_ol={mean_cum_rew_ol:.1f}")
            

        if update_idx >= warmup_updates:
            if ema > best_ema + min_delta:
                best_ema = ema
                bad = 0
                save_checkpoint(os.path.join(args.save_dir, "best.pt"),
                                agent, optim_bandit, optim_motor,
                                extra={"update": update_idx, "ema_score": ema})
            else:
                bad += 1
            if bad >= patience:
                print(f"Early stopping at update {update_idx}, best_ema={best_ema:.2f}")
                break
    


        

                # ---- TensorBoard logging ----
        global_step = update_idx  # one step per update

        # scalar losses
        writer.add_scalar("Loss/ChoiceLoss", var_loss, global_step)
        writer.add_scalar("Loss/MotorLoss", motor_loss, global_step)

        # rewards
        writer.add_scalar("Reward/MeanCumSession_IL", mean_cum_rew_il, global_step)
        writer.add_scalar("Reward/MeanCumSession_AO", mean_cum_rew_ao, global_step)
        writer.add_scalar("Reward/MeanCumSession_OL", mean_cum_rew_ol, global_step)


    ckpt = torch.load(os.path.join(args.save_dir, "best.pt"), map_location=device, weights_only=False)
    print(f"Loaded best checkpoint from update {ckpt['extra']['update']}, ema_score={ckpt['extra']['ema_score']:.2f}")
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()

    eval_il = []
    eval_ao = []
    eval_ol = []

    mean_cum_rew_il = 0.0
    mean_cum_rew_ao = 0.0
    mean_cum_rew_ol = 0.0

    num_eval_sessions = 32  # small but stable

    for i in range(num_eval_sessions):
        res = ray.get(
            workers[0].run_session.remote(
                {k: v.cpu() for k, v in agent.state_dict().items()},
                probs_this_session = [(0.8, 0.2) if np.random.rand() < 0.5 else (0.2, 0.8) for _ in range(session_K)],
                print_this_session=False
            )
        )

        (
            _, _, _,
            _, _, _, _,
            _, _,
            cum_rew_il, cum_rew_ao, cum_rew_ol,
            high_reward_choices_il,
            high_reward_choices_ao,
            high_reward_choices_ol
        ) = res

        # extract from env stats (you already compute these)
        eval_il.append(high_reward_choices_il)  # high_reward_choices_il
        eval_ao.append(high_reward_choices_ao)
        eval_ol.append(high_reward_choices_ol)

        mean_cum_rew_il += cum_rew_il
        mean_cum_rew_ao += cum_rew_ao
        mean_cum_rew_ol += cum_rew_ol

    mean_il = np.array(eval_il).mean(axis=0)
    mean_ao = np.array(eval_ao).mean(axis=0)
    mean_ol = np.array(eval_ol).mean(axis=0)

    mean_cum_rew_il /= num_eval_sessions
    mean_cum_rew_ao /= num_eval_sessions
    mean_cum_rew_ol /= num_eval_sessions

    print(f"Eval results over {num_eval_sessions} sessions:")
    print(f" Mean cum reward IL: {mean_cum_rew_il:.1f}, AO: {mean_cum_rew_ao:.1f}, OL: {mean_cum_rew_ol:.1f}")

    fig, ax = plt.subplots(figsize=(6,4))
    trials = np.arange(session_N)

    ax.plot(trials, mean_il, label="IL")
    ax.plot(trials, mean_ao, label="AO")
    ax.plot(trials, mean_ol, label="OL")

    ax.set_xlabel("Trial index")
    ax.set_ylabel("Mean high-reward choice")
    ax.set_title("Per-trial performance (best checkpoint)")
    ax.legend()
    ax.grid(True)

    plot_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # save to file
    plot_path = os.path.join(plot_dir, "PerTrialReward.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")


    writer.add_figure("Eval/PerTrialReward", fig)
    plt.close(fig)


    ray.shutdown()
    writer.close()


