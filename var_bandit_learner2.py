import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F
import math

class CNNEncoder(nn.Module):
    """A small CNN for processing visual inputs (expects NCHW)."""
    def __init__(self, feature_dim):
        super().__init__()
        # build the CNN feature extractor
        self.cnn = self.__build_cnn(3, feature_dim)

    def __build_cnn(self, in_channels, feature_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((9, 9)),  # ensure output size is (9,9)
            nn.Flatten(),
            nn.Linear(64 * 9 * 9, feature_dim), 
            nn.ReLU()
        )
    
    def forward(self, x):
        x = x.float()
        return self.cnn(x)

def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))

def log_prob_tanh_gaussian(mu, log_std, pretahn_actions):
    # u = atanh(torch.clamp(actions, -1 + 1e-6, 1 - 1e-6))
    std = torch.exp(log_std)
    z = (pretahn_actions - mu) / (std + 1e-6)

    log_prob = -0.5 * (z.pow(2) + 2 * log_std + math.log(2 * math.pi))
    log_prob = log_prob.sum(dim=-1)  # sum over action dimensions

    corr = torch.log(1 - torch.tanh(pretahn_actions).pow(2) + 1e-6).sum(dim=-1)  # correction term
    # print("log_prob.shape:", log_prob.shape, "corr.shape:", corr.shape, "pretanh_actions.shape:", pretahn_actions.shape)
    return log_prob - corr  # (B,)


class BanditLearner(nn.Module):
    def __init__(self, input_size, feature_dim, rnn_hidden_size, action_dim,
                 log_std_min=-5.0, log_std_max=-1.0,
                 num_pairs=None, max_trials=None):
        nn.Module.__init__(self)

        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)

        self.debug_gru_inputs = {"rollout": [], "update": []}


        self.enc = CNNEncoder(feature_dim)

        # --- Bandit RNN + heads
        # input_size = feature_dim + 2 + 1  # features + prev_action(2) + prev_reward(1)
        self.rnn = nn.GRU(input_size=input_size, hidden_size=rnn_hidden_size)

        # choice_inp = rnn_hidden_size
        self.arm_reward_head = nn.Sequential(
            nn.Linear(rnn_hidden_size + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )


        # self.choice_v_head = nn.Sequential(nn.Linear(combined_dim,128), nn.ReLU(), nn.Linear(128,1))

        # NEW: critic for the bandit choice (value at trial start)

        self.mlp_pos = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        # NEW: goal vector g = goal_center - xy (both normalized)
        self.mlp_goal = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )
        
        motor_inp = 2 + 32 + 32  # choice_target + xypos + goalvec
        self.mu_head = nn.Sequential(nn.Linear(motor_inp, 128), nn.ReLU(), nn.Linear(128, action_dim))
        self.log_std_head = nn.Sequential(nn.Linear(motor_inp, 128), nn.ReLU(), nn.Linear(128, action_dim))
        critic_inp = feature_dim + 2 
        # self.value_head = nn.Sequential(nn.Linear(critic_inp, 128), nn.ReLU(), nn.Linear(128, 1))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim


    def encode(self, x):
        # x = self.downsample(x)  # (T, C, H, W)
        return self.enc(x)    

    
    def rnn_fwd(self, left_feats, right_feats, action, reward, meta_ep_start, h):
        """
        features: (T, D_f)      or (T, B, D_f)
        action:   (T, D_a)      or (T, B, D_a)
        reward:   (T, 1)        or (T, B, 1)
        h:        (1, H)        or (1, B, H)

        Returns:
            out:   (T, H)       or (T, B, H)   (same rank as features)
            new_h: (1, H)       or (1, B, H)
        """
        # x = torch.cat([left_feats, right_feats,  action, reward, meta_ep_start], dim=-1).to(left_feats.dtype).to(left_feats.device)
        aL = action[..., 0:1]          # (T,B,1)
        aR = action[..., 1:2]

        chosen_feat   = aL * left_feats  + aR * right_feats
        unchosen_feat = aL * right_feats + aR * left_feats

        x = torch.cat([chosen_feat, unchosen_feat, reward, meta_ep_start], dim=-1).to(left_feats.dtype).to(left_feats.device)


        # Case 1: unbatched old-style input (T, D)
        if x.dim() == 2:
            # add batch dimension B = 1 so GRU sees (T, 1, D)
            x = x.unsqueeze(1)  # (T, 1, D)

            if h is not None and h.dim() == 2:
                # (1, H) -> (1, 1, H)
                h = h.unsqueeze(1)

            out, new_h = self.rnn(x, h)  # out: (T, 1, H), new_h: (1, 1, H)

            # squeeze batch dim back out for compatibility with old code
            out = out.squeeze(1)         # (T, H)
            new_h = new_h.squeeze(1)     # (1, H)
            return out, new_h

        # Case 2: batched input (T, B, D)
        elif x.dim() == 3:
            # h is expected to already be (1, B, H)
            out, new_h = self.rnn(x, h)  # out: (T, B, H)
            return out, new_h

        else:
            raise ValueError(f"Unexpected input rank {x.dim()} for rnn_fwd")

    
    
    def reward_compute(self, h, left_feat, right_feat):
        # h: (T,B,H) or (1,H)
        l = self.arm_reward_head(torch.cat([h, left_feat],  dim=-1)).squeeze(-1)
        r = self.arm_reward_head(torch.cat([h, right_feat], dim=-1)).squeeze(-1)
        return torch.stack([l, r], dim=-1)  # (...,2)



        
    def motor_fwd(self, choice_target, xy_pos=None, goal_vec=None):
        # device = choice_target.device
        # T = features.size(0)

        # if xy_pos is None:  xy_pos  = torch.zeros(T, 2, device=device, dtype=torch.float32)
        # if goal_vec is None: goal_vec = torch.zeros(T, 2, device=device, dtype=torch.float32)

        pos_emb  = self.mlp_pos(xy_pos)       # (T,32)
        goal_emb = self.mlp_goal(goal_vec)    # (T,32)

        # policy input (no h_t): F + 2 + 32 + 32
        motor_inp  = torch.cat([choice_target, pos_emb, goal_emb], dim=-1)

        # critic input (no h_t): F + 2
        # critic_inp = torch.cat([choice_target], dim=-1)

        mu      = self.mu_head(motor_inp)
        log_std = self.log_std_head(motor_inp).clamp(self.log_std_min, self.log_std_max)
        # value   = self.value_head(critic_inp).squeeze(-1)
        return mu, log_std
    

    def bandit_parameters(self):
        # everything that should be updated by bandit_loss
        modules = []

        modules += [
            self.rnn,
            self.arm_reward_head,
            self.enc
            # self.fam_head,   # if you use familiarity as bandit aux
        ]

        for m in modules:
            if m is None:
                continue
            for p in m.parameters():
                yield p



    def motor_parameters(self):
        # everything that should be updated by motor_loss
        modules = [
            self.mlp_pos,
            self.mlp_goal,
            self.mu_head,
            self.log_std_head,
        ]
        for m in modules:
            for p in m.parameters():
                yield p

    def update2(self, optim_bandit, optim_motor, xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
        bandit_obs,           # list of (T, C, H, W) tensors
        chosen_bandits_buf,   # list of (T, 2) tensors
        bandit_rewards_buf,   # list of (T,) tensors
        meta_ep_start_buf,    # list of (T,) tensors
        device,
    ):
        # -----------------------------
        # 1) MOTOR data: keep flat (no batch)
        # -----------------------------
        xy_pos   = torch.as_tensor(np.stack(xy_pos_buf), device=device, dtype=torch.float32)
        goalvec  = torch.as_tensor(np.asarray(goal_vec_buf, np.float32), device=device)
        chosen_bandits_motor = torch.as_tensor(
            np.stack(chosen_bandits_motor_buf), device=device, dtype=torch.float32
        )
        # shapes: (S_total, 2) / (S_total, 2) / (S_total, 2)

        # -----------------------------
        # 2) BANDIT data: batched (B, T, ...)
        # -----------------------------
        # bandit_obs: list of B tensors, each (T, C, H, W)
        left_obs  = torch.stack([torch.tensor(b["left"],  device=device, dtype=torch.float32) for b in bandit_obs], dim=0)  # (B,T,C,H,W)
        right_obs = torch.stack([torch.tensor(b["right"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)
        chosen_bandits = torch.stack(chosen_bandits_buf, dim=0).to(device)  # (B, T, 2)
        rewards_bandits = torch.stack(bandit_rewards_buf, dim=0).to(device) # (B, T)
        meta_ep_start = torch.stack(meta_ep_start_buf, dim=0).to(device)    # (B, T)

        B, T, C, H, W = left_obs.shape

        # Reorder to time-major (T, B, ...)
        chosen_bandits = chosen_bandits.permute(1, 0, 2)           # (T, B, 2)
        rewards_bandits = rewards_bandits.permute(1, 0)            # (T, B)
        meta_ep_start   = meta_ep_start.permute(1, 0)              # (T, B)

        # For GRU:
        rewards_rnn = rewards_bandits.unsqueeze(-1)    # (T, B, 1)
        start_rnn   = meta_ep_start.unsqueeze(-1)      # (T, B, 1)

        num_epochs = 1

        var_loss_sum, var_slices = 0.0, 0
        motor_loss_sum, motor_slices = 0.0, 0

        for _ in range(num_epochs):

            optim_motor.zero_grad(set_to_none=True)
            optim_bandit.zero_grad(set_to_none=True)

            # -----------------------------
            # 3) BANDIT loss
            # -----------------------------
            h_rnn = torch.zeros(1, B, self.rnn.hidden_size, device=device)

            left_flat  = left_obs.reshape(B*T, C, H, W)
            right_flat = right_obs.reshape(B*T, C, H, W)

            left_feats  = self.encode(left_flat).view(B, T, -1).permute(1, 0, 2)   # (T,B,F)
            right_feats = self.encode(right_flat).view(B, T, -1).permute(1, 0, 2)  # (T,B,F)


            rnn_out, h_rnn = self.rnn_fwd(
                left_feats,       # (T, B, F)
                right_feats,      # (T, B, F)
                chosen_bandits,     # (T, B, 2)
                rewards_rnn,        # (T, B, 1)
                start_rnn,          # (T, B, 1)
                h_rnn               # (1, B, H)
            )                       # rnn_out: (T, B, H)

            # history one step back
            h_seq_orig = torch.zeros_like(rnn_out)
            h_seq_orig[1:] = rnn_out[:-1]          # (T, B, H)

            reward_logits = self.reward_compute(h_seq_orig, left_feats, right_feats)  # (T, B, 2)
            left_logits   = reward_logits[..., 0]  # (T, B)
            right_logits  = reward_logits[..., 1]  # (T, B)

            p_left_rwd  = torch.distributions.Bernoulli(logits=left_logits)
            p_right_rwd = torch.distributions.Bernoulli(logits=right_logits)

            logp_left  = p_left_rwd.log_prob(rewards_bandits)   # (T, B)
            logp_right = p_right_rwd.log_prob(rewards_bandits)  # (T, B)

            logp_rewards = torch.where(
                chosen_bandits[..., 0] == 1,   # left chosen?
                logp_left,
                logp_right
            )                                  # (T, B)

            var_logp_loss = logp_rewards.mean()

            # KL to Bernoulli(0.5) prior, per arm
            eps = 1e-8
            probs = torch.sigmoid(reward_logits)  # (T, B, 2)
            prior_p = 0.5

            var_kl_loss = (
                probs * (torch.log(probs + eps) - math.log(prior_p)) +
                (1.0 - probs) * (torch.log(1.0 - probs + eps) - math.log(1.0 - prior_p))
            ).sum(dim=-1).mean()  # sum over arms, mean over T,B

            # variational_loss = -var_logp_loss + 0.01 * var_kl_loss
            variational_loss = -var_logp_loss
            variational_loss.backward()

            var_loss_sum += variational_loss.item()
            var_slices += 1

            # -----------------------------
            # 4) MOTOR loss (unchanged, flat over all steps)
            # -----------------------------
            mini_batch_size = 16384  # Tune based on your GPU; smaller = less mem, but slower
            total_steps = len(xy_pos_buf)
            motor_loss_sum += 0.0  # Already in your code; keep for averaging

            for start in range(0, total_steps, mini_batch_size):
                end = min(start + mini_batch_size, total_steps)
                
                # Slice tensors
                xy_slice = xy_pos[start:end]
                goal_slice = goalvec[start:end]
                chosen_slice = chosen_bandits_motor[start:end]
                
                # Forward on slice
                mu, log_std = self.motor_fwd(
                    chosen_slice,  # no credit to choice through motor
                    xy_slice,
                    goal_slice
                )
                
                # Target computation (same as before, but on slice)
                dist = goal_slice.norm(dim=-1, keepdim=True) + 1e-6
                g_hat = goal_slice / dist
                speed = (dist / math.sqrt(8.0)).clamp(0.0, 1.0)
                target = (g_hat * speed).clamp(-0.999, 0.999)
                u_target = atanh(target)
                
                L_reach = F.mse_loss(mu, u_target)
                motor_loss_mini = L_reach
                
                # Backward on mini-loss (grads accumulate)
                motor_loss_mini.backward()
                
                motor_loss_sum += motor_loss_mini.item() * ((end - start) / total_steps)  # Weighted avg for logging

            # -----------------------------
            # 5) Optim step
            # -----------------------------
            torch.nn.utils.clip_grad_norm_(list(self.bandit_parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(list(self.motor_parameters()), 1.0)

            optim_bandit.step()
            optim_motor.step()

        # -----------------------------
        # 6) Return averaged losses
        # -----------------------------
        avg_var_loss   = var_loss_sum   / max(1, var_slices)
        avg_motor_loss = motor_loss_sum / max(1, motor_slices)
        return avg_var_loss, avg_motor_loss
