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
import var_bandit_learner2 as bl
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

    def log(*args, **kwargs):
        if print_this_session and worker_id == 0:
            print(*args, **kwargs)

    # h_rnn = torch.zeros(1, agent.rnn.hidden_size, device=device)  # GRU hidden state
    q_left_tokens = []   # past pair keys (1,H)
    q_right_tokens = []   # past pair keys (1,H)
    k_tokens = []   # past keys (1,H)
    o_tokens = []   # past outcome values (1,H)
    q_this_trial = None


    obs, info = env.reset()
    done = False

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

            curr_trial_condition = info.get("curr_trial_condition")

            left_feats = agent.encode(left_view)    # (1,F)
            right_feats = agent.encode(right_view)  # (1,F)

            left_obs.append(left_view.squeeze(0).detach().cpu().numpy())
            right_obs.append(right_view.squeeze(0).detach().cpu().numpy())

            # Build query token (swap-invariant) and attend over history
            # Build query token (swap-invariant) and attend over history
            q_left = agent.q_in(left_feats)    # (T,B,H)
            q_right = agent.q_in(right_feats)  # (T,B,H)

            Tpast = len(o_tokens)
            if Tpast == 0:
                ctx_left = agent.attn_ln(q_left.unsqueeze(0))
                ctx_right = agent.attn_ln(q_right.unsqueeze(0))
            else:
                # query is CURRENT token (len=1)
                ql = q_left.unsqueeze(0)           # (1,1,H)
                qr = q_right.unsqueeze(0)          # (1,1,H)

                # keys/values are PAST tokens
                k = torch.stack(k_tokens, dim=0)   # (Tpast,1,H)
                v = torch.stack(o_tokens, dim=0)   # (Tpast,1,H)

                # print("K shape:", k.shape, "V shape:", v.shape)

                ctx_left,  _ = agent.attn(ql, k, v)
                ctx_right, _ = agent.attn(qr, k, v)

                ctx_left  = agent.attn_ln(ctx_left  + ql)
                ctx_right = agent.attn_ln(ctx_right + qr)


            if(curr_trial_condition != "AO-o" and curr_trial_condition != "OL-o"):

                left_logits  = agent.reward_compute(ctx_left,  left_feats.unsqueeze(0))   # (1,1,F)
                right_logits = agent.reward_compute(ctx_right, right_feats.unsqueeze(0))

                tau = 0.2  # temperature: smaller => more decisive; try 0.05–1.0

                action_logits = torch.cat([left_logits, right_logits], dim=-1)  # (1,1,2)

                # Option A: probs (nice for logging)
                p_choose = torch.softmax(action_logits / tau, dim=-1)           # (1,1,2)
                p_left  = p_choose[0, 0, 0]
                p_right = p_choose[0, 0, 1]

                # log("p_left:", p_left, "p_right:", p_right)

                # Sample action from categorical distribution
                dist = torch.distributions.Categorical(probs=p_choose[0, 0])    # (2,)
                a_t = dist.sample()                                            # scalar: 0=left, 1=right

                choice_target = F.one_hot(a_t, num_classes=2).unsqueeze(0).float()  # (1,2)
            else:
                # observational trial: no choice, use env-chosen action
                a_t = 0  # Dummy
                choice_target = F.one_hot(torch.tensor(a_t), num_classes=2).unsqueeze(0).float()  # (1,2)
                log("Obs trial: env chose ", "left" if a_t == 0 else "right")    


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

            actor = 0
            if curr_trial_condition == "AO-o" or curr_trial_condition == "OL-o":
                # on obs trials, use env-chosen action
                a_t = info.get("selected_target", -1)
                choice_target = F.one_hot(torch.tensor(a_t), num_classes=2).unsqueeze(0).float()  # (1,2)
                actor = 1
            if curr_trial_condition == "AO-o":
                reward = 2    
            # rnn_fwd returns RAW GRU out for this trial (outcome token)
            # o_t, h_rnn = agent.rnn_fwd(left_feats, right_feats, a_oh, r_t, meta_ep_start_torch, h_rnn)  # o_t: (1,H)

            aL = choice_target[..., 0:1]
            aR = choice_target[..., 1:2]

            chosen_feat   = aL * left_feats  + aR * right_feats
            unchosen_feat = aL * right_feats + aR * left_feats

            # x = r_t.to(left_feats.dtype)
            action_emb = agent.action(torch.tensor(a_t, dtype=torch.long).unsqueeze(0))  # (1,H)
            actor_emb = agent.actor(torch.tensor(actor, dtype=torch.long).unsqueeze(0))      # (1,H)
            reward_emb = agent.rwd_in(torch.tensor(reward, dtype=torch.long).unsqueeze(0))      # (1,H)
            x = action_emb + actor_emb + reward_emb


            o_tokens.append(x)
            q_left_t = agent.q_in(left_feats)    # (1,H)
            q_right_t = agent.q_in(right_feats)  # (1,H)
            q_left_tokens.append(q_left_t)  
            q_right_tokens.append(q_right_t)
            #pair_key = q_left_t + q_right_t  # (1,H)
            pair_key = agent.q_in(chosen_feat)  # (1,H)
            k_tokens.append(pair_key)

            pair_index_ep = info.get("prev_pair_index_in_session", -1)
            
            selected_high_reward = info.get("selected_high_reward_this_trial", -1)
            flipped = info.get("side_is_flipped", False)

            if(curr_trial_condition == "IL"):
                pair_index_counter_il[pair_index_ep] += 1
            elif(curr_trial_condition == "AO-s"):
                pair_index_counter_ao[pair_index_ep] += 1
            elif(curr_trial_condition == "OL-s"):
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
            actor_buf.append(actor)
            trial_cond_buf.append(curr_trial_condition)

            choice_target = None
            ep_start_flag = 1.0

        t += 1

    log("trial_conditions:", env.unwrapped.trial_cond)

    # session stats
    high_reward_choice_count_on_left = info.get("high_reward_choice_count_on_left", -1)
    high_reward_choice_count_on_right = info.get("high_reward_choice_count_on_right", -1)
    total_left_choices = info.get("total_left_choices", -1)
    total_right_choices = info.get("total_right_choices", -1)

    log(
        "Session finished. High-reward choices for il:", high_reward_choices_il,
        ", High-reward choices for ao:", high_reward_choice_ao,
        ", High-reward choices for ol:", high_reward_choice_ol,
        ", High-reward choices on left:", high_reward_choice_count_on_left,
        ", on right:", high_reward_choice_count_on_right,
        ", Total left choices:", total_left_choices,
        ", Total right choices:", total_right_choices,
    )


    bandit_rewards_arr = np.asarray(bandit_rewards_buf, dtype=np.float32)
    trial_cond_arr = np.asarray(trial_cond_buf)

    cum_rewards_il = float(bandit_rewards_arr[trial_cond_arr == "IL"].sum())
    cum_rewards_ao = float(bandit_rewards_arr[trial_cond_arr == "AO-s"].sum())
    cum_rewards_ol = float(bandit_rewards_arr[trial_cond_arr == "OL-s"].sum())

    high_reward_choices_il = np.array(high_reward_choices_il, dtype=np.int32)
    high_reward_choices_ao = np.array(high_reward_choice_ao, dtype=np.int32)
    high_reward_choices_ol = np.array(high_reward_choice_ol, dtype=np.int32)

    obs_bandit = {
        "left":  np.stack(left_obs,  axis=0),
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
            randomize_sides=True,
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
        
    def collect_teacher_data(self, agent_state_dict, probs_this_session, print_this_session=False):
        # implement if needed
        hidden_size = 128
        feature_dim = 128
        input_size = feature_dim + 1

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
    patience = 30
    min_delta = 0.1
    bad = 0
    warmup_updates = 900



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
    # ray.init(ignore_reinit_error=True)

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

                # probs_this_session = [(0.8, 0.2) if np.random.rand() < 0.5 else (0.2, 0.8) for _ in range(session_K)]
                probs_this_session = [(0.8, 0.2)] * session_K ## Sanity check
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


    ckpt = torch.load(os.path.join(args.save_dir, "best.pt"), map_location=device)
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
                probs_this_session=[(0.8, 0.2)] * session_K,
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


