from typing import Any, Tuple, Dict
import time
import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F
import argparse


import visual_bandit_env2 as vbe
import var_bandit_learner as bl
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import math
import pygame
import os
from datetime import datetime

from gymnasium.wrappers import RecordVideo

import math, contextlib

torch.backends.cudnn.enabled = False
torch.autograd.set_detect_anomaly(True)


def save_checkpoint(path, agent, optimizer, extra):
    ckpt = {
        "model_state": agent.state_dict(),
        "optim_state": optimizer.state_dict(),
        "extra": extra,
        "torch": torch.__version__,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path, agent, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    agent.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt.get("extra", {})




                    
def meta_ep_rollout(env, agent, device, obs_buf, pretanh_buf, act_buf, rew_buf, ep_start_buf, done_buf, choice_target_buf, session_K, session_N):

    h_rnn = torch.zeros(1, agent.rnn.hidden_size, device=device)  # initial hidden state
    # for episode in range(session_K*session_N):
    obs,info = env.reset()
    # print("Total episodes in session:", info.get("total_trials_in_session", -1))
    done = False
    ep = 0
    

    pair_index_counter = np.ones(session_K, dtype=np.int32)*-1 # counts per pair
    high_reward_choice_per_N = np.zeros(session_N, dtype=np.int32) # counts per N

    # prev_action = torch.zeros((1, agent.action_dim), dtype=torch.float32, device=device)  # assuming action_dim=2
    prev_bandit_reward = torch.tensor([[0.0]], dtype=torch.float32, device=device)
    prev_bandit_act = torch.zeros((1, 2), dtype=torch.float32, device=device)  # one-hot for prev bandit choice
    ep_start_flag = 1.0  # Indicates the start of a new episode
    # choice_inp_ep_start = None # initial choice input is None, computed at t=0, then constant until t=T
    # choice_logit_ep_start = None # initial choice prob is None, computed at t=0, then constant until t=T
    choice_target = None # initial choice target is None, computed at t=0, then constant until t=T
    episode_timeout = np.empty((0,), dtype=np.float32)
    ep_len = 0
    info_buf = []
    bandit_rewards_buf = []
    xy_pos_buf = []
    goal_vec_buf = []
    choice_logp_buf = []
    seen_before_buf = []
    feats_buf = []
    # rnn_outputs = torch.empty((0, agent.gru.hidden_size), dtype=torch.float32, device=device)
    logp = 0.0
    t = 0
    while not done:
        obs_tensor = (
            torch.as_tensor(obs, device=device).permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)
        )
        small = agent.downsample(obs_tensor)
        if agent.feature_source == "vision":
            features = agent.encode(small).squeeze(0)
                 # (F,) for this step
        ep_start = torch.tensor([[ep_start_flag]], device=device, dtype=torch.float32).squeeze(0)  # (1,) -> we'll expand below


        # CHOICE at episode start
        if ep_start_flag == 1.0:
                # build single-step tensors as (1, D)
            # f1  = features.unsqueeze(0)        # (1, F)
            es1 = ep_start.unsqueeze(0)        # (1, 1)
            if agent.feature_source == "ids":
                # NB: make sure these are 0-based; if your env is 1-based, subtract 1.
                pair_idx_now  = info.get("pair_index_in_session", -1)
                trial_idx_now = int(info.get("trial_index", -1))  # make 0-based

                pair_idx_t  = torch.tensor([pair_idx_now],  device=device, dtype=torch.long)
                trial_idx_t = torch.tensor([trial_idx_now], device=device, dtype=torch.long)

                # print("At trial start: pair_idx_now =", pair_idx_now, ", trial_idx_now =", trial_idx_now, ", ep=", ep)
                f1 = agent.make_ctx_from_ids(pair_idx_t)   # (1, F)
            else:
                small = agent.downsample(obs_tensor)
                f1 = agent.encode(small)                                # (1, F)

            
            # --- Bandit forward for choice ---
            
            # rnn_out, h = agent.(f1, prev_bandit_act, prev_bandit_reward, es1, h)  # rnn_out: (1, H)
            # --- Bandit forward for choice (Thompson sampling with Bernoulli head) ---

            # 1) Get logits from the bandit head using current RNN state
            reward_logits = agent.reward_compute(h_rnn).squeeze(0)  # (2,) -> [logit_left, logit_right]

            # 2) Convert logits -> probabilities
            probs = torch.sigmoid(reward_logits)  # (2,), in (0,1)
            p_left, p_right = probs[0], probs[1]

            # 3) Build an approximate Beta posterior around each p
            #    concentration controls how "confident" the posterior is.
            concentration = 5.0  # hyper-parameter, tune if needed
            alpha_left  = p_left  * concentration
            beta_left   = (1.0 - p_left)  * concentration
            alpha_right = p_right * concentration
            beta_right  = (1.0 - p_right) * concentration

            dist_left  = torch.distributions.Beta(alpha_left,  beta_left)
            dist_right = torch.distributions.Beta(alpha_right, beta_right)

            # 4) Thompson sample & pick the best arm
            u_left  = dist_left.rsample()
            u_right = dist_right.rsample()

            samples = torch.stack([u_left, u_right])           # (2,)
            a_t = torch.argmax(samples)                        # index of chosen arm
            choice_target = F.one_hot(a_t, num_classes=2).unsqueeze(0)  # (1,2)

            ep += 1


        # --- Motor forward (now we have choice_target) ---
        # Build (x,y) in [-1,1]
        W, H = env.unwrapped.W, env.unwrapped.H
        x_pix, y_pix = env.unwrapped.cursor
        xy_norm = np.array([(x_pix/(W-1))*2-1, (y_pix/(H-1))*2-1], dtype=np.float32)

        # Panel centers in pixels → normalize to [-1,1]
        left_c  = np.array([env.unwrapped.left_rect.centerx,  env.unwrapped.left_rect.centery],  np.float32)
        right_c = np.array([env.unwrapped.right_rect.centerx, env.unwrapped.right_rect.centery], np.float32)
        left_c_norm  = np.array([(left_c[0]/(W-1))*2-1,  (left_c[1]/(H-1))*2-1],  np.float32)
        right_c_norm = np.array([(right_c[0]/(W-1))*2-1, (right_c[1]/(H-1))*2-1], np.float32)

        # Chosen center based on one-hot
        chosen_center = left_c_norm if choice_target.argmax(dim=-1).item() == 0 else right_c_norm
        g_norm = chosen_center - xy_norm  # vector *to* goal

        xy_pos_t  = torch.as_tensor(xy_norm).unsqueeze(0).to(device)  # (1,2)
        goal_vec_t= torch.as_tensor(g_norm ).unsqueeze(0).to(device)  # (1,2)

        mu, log_std = agent.motor_fwd(choice_target, xy_pos = xy_pos_t, goal_vec=goal_vec_t)

        # store for training
        xy_pos_buf.append(xy_norm)     # (2,)
        goal_vec_buf.append(g_norm)    # (2,)
    
        std = torch.exp(log_std)
        y = mu + std * torch.randn_like(std)     # pre-tanh noise
        action = torch.tanh(y)
        action_np = action.squeeze(0).detach().cpu().numpy()

        info_buf.append(info)  # store info for training
        next_obs, reward, term, _, info = env.step(action_np)
        ep_len += 1
        done = term
        
        bandit_reward_t = reward

        pair_idx_now = info.get("pair_index_in_session", -1)

        if ep_start_flag == 1.0 and pair_idx_now != -1:
            # true at the first step of the trial; before incrementing counters
            seen_before_flag = float(pair_index_counter[pair_idx_now] >= 0)
        else:
            seen_before_flag = 0.0  # we only supervise at starts

        seen_before_buf.append(seen_before_flag)

        feats_buf.append(f1.squeeze(0).detach().cpu().numpy())  # (F,)
        small_np = small.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        obs_buf.append(small_np.astype(np.uint8))
        pretanh_buf.append(y.squeeze(0).detach().cpu().numpy())
        act_buf.append(action_np.astype(np.float32))
        ep_start_buf.append(ep_start_flag)
        choice_target_buf.append(choice_target.squeeze(0).detach().cpu().numpy())
        done_buf.append(False)
        bandit_rewards_buf.append(bandit_reward_t)
        choice_logp_buf.append(0)

        # xy_pos_buf.append(xy_norm.astype(np.float32))   # (2,)


        if info.get("trial_ended"):
            
            #Apply reward and logp correction for bandit choice
            if choice_target.argmax(dim=-1).item() != info.get("selected_target"):
                reward = 0
            else:
                done_buf[t] = True    
                a_oh        = choice_target                                           # (1, 2) one-hot choice
                r_t         = torch.tensor([[float(reward)]], device=device)          # corrected 0/1
                value_rnn_out, h_rnn       = agent.rnn_fwd(f1, a_oh, r_t, h_rnn)           # update GRU memory
                # rnn_outputs = torch.cat((rnn_outputs, rnn_out.detach()), dim=0)  # (T, H)
            
            pair_index_ep = info.get("pair_index_in_session", -1)
            pair_index_counter[pair_index_ep] += 1    
            selected_high_reward = info.get("selected_high_reward_this_trial", False)
            if(selected_high_reward):
                    # print("High reward choice made for pair index", pair_index_ep, "at count", pair_index_counter[pair_index_ep])
                    high_reward_choice_per_N[pair_index_counter[pair_index_ep]] += 1
            print("Ep: ", info.get("trial_index") , ", idx = ", pair_idx_now, ", choice target:", choice_target.argmax(dim=-1).item(), ", Reached Target:", info.get("selected_target"), ", reward:", reward, ", selected_high_reward = ", selected_high_reward, ", p_left =", p_left.item(), ", p_right =", p_right.item())    
            
            ep_start_flag = 1.0
            # print("Trial ended. t=", t, ", pair_index_ep=", pair_index_ep, ", pair_idx_t=", pair_idx_t,  ", choice_target:", choice_target, "reward:", reward, ", done_buf[t]=", done_buf[t])

            choice_target = None
            # prev_action.zero_()
            # prev_reward.zero_()
            mask = np.zeros(ep_len, dtype=np.float32) if info.get("timeout") else np.ones(ep_len, dtype=np.float32)
            episode_timeout = np.concatenate((episode_timeout, mask), axis=0)
            ep_len = 0  
            # ep = 0
 
    
        else:
            ep_start_flag = 0.0

        rew_buf.append(float(reward))
        

        obs = next_obs
        t += 1
        # prev_action = action
        # prev_reward = torch.as_tensor([[reward]], dtype=torch.float32, device=device)

    print("Done with rollout at t =", t, ", feats_buf len =", len(feats_buf), ", rew_buf len =", len(rew_buf), ", act_buf len =", len(act_buf), ", done_buf len =", len(done_buf))
    high_reward_choice_count_on_left = info.get("high_reward_choice_count_on_left", -1)
    high_reward_choice_count_on_right = info.get("high_reward_choice_count_on_right",-1)
    total_left_choices = info.get("total_left_choices",-1)
    total_right_choices = info.get("total_right_choices",-1)
    print("Session finished. High-reward choices per N:", high_reward_choice_per_N, ", High-reward choices on left:", high_reward_choice_count_on_left, ", on right:", high_reward_choice_count_on_right, ", Total left choices:", total_left_choices, ", Total right choices:", total_right_choices, ", Total trials in session:", len(rew_buf))
    high_reward_choice_per_N = high_reward_choice_per_N.astype(float)
    high_reward_choice_per_N /= float(session_K)  # normalize by K
    high_reward_choice_per_N = high_reward_choice_per_N.round(2)

    # if info.get('probe_fam_inv_err'):
    #     print(f"probe: fam_inv_err={np.mean(info['probe_fam_inv_err']):.3g}, "
    #       f"flip_consistency={np.mean(info['probe_flip_consistency']):.3g}")

    
    return obs_buf, pretanh_buf, act_buf, rew_buf, bandit_rewards_buf, ep_start_buf, xy_pos_buf, goal_vec_buf, done_buf, choice_target_buf, choice_logp_buf, episode_timeout, info, high_reward_choice_per_N, info_buf, seen_before_buf, feats_buf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)


    session_K = 1
    session_N = 20

    
    hidden_size=128
    feature_dim = 128
    input_size = feature_dim + 2 + 1  # feature_dim + prev_action_dim + prev_reward_dim + ep_start_dim
    # agent = bl.BanditLearner(input_size=input_size, feature_dim= feature_dim, rnn_hidden_size=hidden_size, action_dim=2)
    agent = bl.BanditLearner(
        input_size=input_size,
        feature_dim=feature_dim,
        rnn_hidden_size=hidden_size,
        action_dim=2,
        feature_source="ids",           # <— NEW
        num_pairs=session_K,            # <— NEW (vocab for pair indices)
        max_trials=session_N*session_K            # <— NEW (vocab for trial indices)
    )

    optim_bandit = torch.optim.Adam(agent.bandit_parameters(), lr=1e-3)
    optim_motor = torch.optim.Adam(agent.motor_parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    past_cnn_features = []
    
    # pair_probs = [(0.8, 0.2) if i % 2 == 1 else (0.2, 0.8) for i in range(500)]
    env = vbe.TwoChoiceReachingEnv(
            W = 384,
            H = 400,
            render_mode="rgb_array",
            seed=0,
            session_K=session_K,
            session_N=session_N,
            trial_ms=3000,
            randomize_sides=False,
        )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    video_dir = os.path.join("videos", f"run_{timestamp}")

    # record every episode into ./videos (needs ffmpeg on your system)
    # env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: ep % 20 == 0)
    # env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True)


    reach_ema = 0.0

    for ses in range(500):

        K = env.unwrapped.session_K
        if ses % 2 == 0:
            probs_this_session = [(0.8, 0.2)] * K
        else:
            probs_this_session = [(0.2, 0.8)] * K

        # rng = np.random.default_rng(seed=ses + args.seed * 1000)
        # L = np.round(rng.uniform(0.1, 0.9, size=K),3)
        # R = np.round(1-L, 3)
        # probs_this_session = [(L[i], R[i]) for i in range(K)]

        env.unwrapped.pair_probs = probs_this_session    

        # env.unwrapped.pair_probs= [ (0.8,0.2) if i % 2 == 0 else (0.2,0.8) for i in range(K)]
        # env.unwrapped.pair_probs = [
        #     (0.8, 0.2) if np.random.rand() < 0.5 else (0.2, 0.8)
        #     for _ in range(session_K)
        # ]



        probs_this_session = env.unwrapped.pair_probs

        print(f"\n=== Starting session {ses+1} with pair probabilities: {probs_this_session} ===")
             
        # h = torch.zeros(1, 1, hidden_size).to(device)  # initial hidden state
        

        obs_buf, pretanh_buf, act_buf, rew_buf, bandit_rewards_buf, ep_start_buf, done_buf, choice_target_buf = [], [], [], [], [], [], [], []
        # agent.apply_param_noise()
        (obs_buf, pretanh_buf, act_buf, rew_buf, bandit_rewards_buf, 
         ep_start_buf, xy_pos_buf, goal_vec_buf, done_buf, 
         choice_target_buf, choice_logp_buf, episode_timeouts, info, 
         high_reward_choice_per_N, info_buf, seen_before_buf, feats_buf) = meta_ep_rollout(env, agent, device, 
                                                                                   obs_buf, pretanh_buf, 
                                                                                   act_buf, rew_buf, 
                                                                                   ep_start_buf, done_buf, 
                                                                                   choice_target_buf, session_K, 
                                                                                   session_N)
        print(
            f"Session {ses+1}: reward_sum={info['cum_session_rewards']:.1f}, "
            f"truncs={info['cum_session_truncations']}, "
            f"terms={info['cum_session_terminations']}, "
            f"trials={info['total_trials_in_session']}, "
            f"high_reward_choices={info['cum_session_high_reward_choices']}, "
            f"flips={info.get('flips_this_episode', 0)}, ",
            f"rew_buf sum={sum(rew_buf):.1f}, ",
            f'high_reward_choice_per_N={high_reward_choice_per_N}'
        )


        # if sum(rew_buf) != 0:
        loss, policy_loss = agent.update2(
                                optim_bandit, optim_motor,      # pass both optimizers
                                rew_buf, choice_target_buf,
                                xy_pos_buf, goal_vec_buf, done_buf, feats_buf, device
                            )

        print(f"Choice Loss: {loss:.4f}, Motor Loss: {policy_loss:.4f}")



        with torch.no_grad():
            if high_reward_choice_per_N[-3:].mean() > 0.7:  # Last 3 trials
                print("✓ Agent is exploiting!")
            else:
                print("✗ Still exploring/random")


        
        # if (ses + 1) % 10 == 0:  # every 10 sessions
        #     stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        #     save_dir = getattr(args, "save_dir", "checkpoints")  # if you add CLI arg
        #     ckpt_path = os.path.join(save_dir, f"visrnn_ses{ses+1}_{stamp}.pt")
        #     extra = {
        #         "session": ses + 1,
        #         "env_info": info,  # last session stats
        #         "feature_dim": feature_dim,
        #         "hidden_size": hidden_size,
        #         "action_dim": 2,
        #     }
            # save_checkpoint(ckpt_path, agent, optimizer, extra)
            # print(f"Saved checkpoint to {ckpt_path}")

        # print(f"Updated param noise std: {new_std:.5f}")
        




    env.close()