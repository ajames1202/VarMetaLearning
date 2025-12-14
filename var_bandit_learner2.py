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
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, feature_dim),
        )

    def forward(self, x):
        # x: (N,C,H,W)
        return self.cnn(x)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def log_prob_tanh_normal(action, mean, log_std, eps=1e-6):
    """Log prob under tanh-squashed Normal with reparam correction."""
    std = torch.exp(log_std)
    # undo tanh
    pretahn_actions = torch.atanh(action.clamp(-1 + eps, 1 - eps))
    # Normal log prob
    normal = torch.distributions.Normal(mean, std)
    log_prob = normal.log_prob(pretahn_actions).sum(dim=-1)
    # correction
    corr = torch.log(1 - torch.tanh(pretahn_actions).pow(2) + 1e-6).sum(dim=-1)
    return log_prob - corr


class BanditLearner(nn.Module):
    def __init__(self, input_size, feature_dim, rnn_hidden_size, action_dim,
                 log_std_min=-5.0, log_std_max=-1.0,
                 num_pairs=None, max_trials=None):
        nn.Module.__init__(self)

        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)

        self.debug_gru_inputs = {"rollout": [], "update": []}

        self.enc = CNNEncoder(feature_dim)

        # --- Bandit RNN + heads
        self.attn = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, kdim=rnn_hidden_size, vdim=input_size
)
        
        self.q_in = nn.Linear(feature_dim, rnn_hidden_size, bias=False)

        self.arm_reward_head = nn.Sequential(
            nn.Linear(rnn_hidden_size + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # motor policy
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mlp_pos = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        self.mlp_goal = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        motor_in = action_dim + 32 + 32
        self.mu_head = nn.Sequential(
            nn.Linear(motor_in, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(motor_in, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )

    def bandit_parameters(self):
        # everything that should be updated by bandit_loss
        modules = []
        modules += [
            self.attn,
            self.q_in,
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
        modules = [
            self.mlp_pos,
            self.mlp_goal,
            self.mu_head,
            self.log_std_head,
        ]
        for m in modules:
            if m is None:
                continue
            for p in m.parameters():
                yield p

    def encode(self, x):
        # x = self.downsample(x)  # (T, C, H, W)
        return self.enc(x)

    
    def reward_compute(self, h, feats):
        # h: (T,B,H) or (1,H)
        l = self.arm_reward_head(torch.cat([h, feats],  dim=-1))
        return l  # (...,2)

    def motor_fwd(self, choice_target, xy_pos=None, goal_vec=None):
        pos_emb  = self.mlp_pos(xy_pos)
        goal_emb = self.mlp_goal(goal_vec)
        motor_inp  = torch.cat([choice_target, pos_emb, goal_emb], dim=-1)
        mu      = self.mu_head(motor_inp)
        log_std = self.log_std_head(motor_inp).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def update2(self, optim_bandit, optim_motor, xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
        bandit_obs,           # list of dicts with "left"/"right"
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

        # -----------------------------
        # 2) BANDIT data: batched (B, T, ...)
        # -----------------------------
        left_obs  = torch.stack([torch.tensor(b["left"],  device=device, dtype=torch.float32) for b in bandit_obs], dim=0)  # (B,T,C,H,W)
        right_obs = torch.stack([torch.tensor(b["right"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)  # (B,T,C,H,W)
        chosen_bandits  = torch.stack(chosen_bandits_buf, dim=0).to(device)     # (B,T,2)
        rewards_bandits = torch.stack(bandit_rewards_buf, dim=0).to(device)     # (B,T)
        meta_ep_start   = torch.stack(meta_ep_start_buf, dim=0).to(device)      # (B,T)

        B, T, C, H, W = left_obs.shape

        # time-major
        chosen_bandits  = chosen_bandits.permute(1, 0, 2)   # (T,B,2)
        rewards_bandits = rewards_bandits.permute(1, 0)     # (T,B)
        meta_ep_start   = meta_ep_start.permute(1, 0)       # (T,B)

        rewards_rnn = rewards_bandits.unsqueeze(-1)  # (T,B,1)
        start_rnn   = meta_ep_start.unsqueeze(-1)    # (T,B,1)

        num_epochs = 1
        var_loss_sum, var_slices = 0.0, 0
        motor_loss_sum, motor_slices = 0.0, 0

        for _ in range(num_epochs):
            optim_motor.zero_grad(set_to_none=True)
            optim_bandit.zero_grad(set_to_none=True)

            # -----------------------------
            # 3) BANDIT loss (query/outcome interleaving)
            # -----------------------------

            left_flat  = left_obs.reshape(B*T, C, H, W)
            right_flat = right_obs.reshape(B*T, C, H, W)

            left_feats  = self.encode(left_flat).view(B, T, -1).permute(1, 0, 2)   # (T,B,F)
            right_feats = self.encode(right_flat).view(B, T, -1).permute(1, 0, 2)  # (T,B,F)

            
            aL = chosen_bandits[..., 0:1]
            aR = chosen_bandits[..., 1:2]

            chosen_feat   = aL * left_feats  + aR * right_feats
            unchosen_feat = aL * right_feats + aR * left_feats

            x = torch.cat([chosen_feat, unchosen_feat, rewards_rnn, start_rnn], dim=-1).to(left_feats.dtype)




            # # Query tokens from current observation (swap-invariant)
            q_left_all  = self.q_in(left_feats)     # (T,B,H)
            q_right_all = self.q_in(right_feats)    # (T,B,H)
            pair_key_all = q_left_all + q_right_all # (T,B,H)

            q_left_q  = q_left_all[1:]              # queries at t=1..T-1
            q_right_q = q_right_all[1:]

            k_hist = pair_key_all[:-1]              # keys at t=0..T-2  
            v_hist = x[:-1]                   # vals at t=0..T-2  

            L = q_left_q.size(0)  # query length (T-1)

            # mask future keys: shape (L, L), True means "masked"
            attn_mask = torch.triu(
                torch.ones((L, L), device=q_left_q.device, dtype=torch.bool),
                diagonal=1
            )


            ctx_l, _ = self.attn(q_left_q,  k_hist, v_hist, attn_mask=attn_mask)
            ctx_r, _ = self.attn(q_right_q, k_hist, v_hist, attn_mask=attn_mask)
            # t=0 has no history, so we just use the local pair token
            ctx_left = ctx_l.new_zeros(T, B, q_left_all.size(-1)) #
            ctx_left[0]  = q_left_all[0]
            ctx_right = ctx_r.new_zeros(T, B, q_right_all.size(-1)) # (T,B,H)
            ctx_right[0] = q_right_all[0]

            ctx_left[1:]  = ctx_l      
            ctx_right[1:] = ctx_r      


            left_logits = self.reward_compute(ctx_left, left_feats)  # (T,B,1)
            right_logits = self.reward_compute(ctx_right, right_feats)  # (T,B,1)


            p_left_rwd  = torch.distributions.Bernoulli(logits=left_logits)
            p_right_rwd = torch.distributions.Bernoulli(logits=right_logits)

            
            logp_left  = p_left_rwd.log_prob(rewards_rnn)
            logp_right = p_right_rwd.log_prob(rewards_rnn)

            logp_rewards = torch.where(
                chosen_bandits[..., 0] == 1,
                logp_left,
                logp_right
            )

            var_logp_loss = logp_rewards.mean()


            # (optional) KL term kept from your code (unused in final loss unless you re-enable it)
            eps = 1e-8
            # probs = torch.sigmoid(reward_logits)
            # prior_p = 0.5
            # var_kl_loss = (
            #     probs * (torch.log(probs + eps) - math.log(prior_p)) +
            #     (1.0 - probs) * (torch.log(1.0 - probs + eps) - math.log(1.0 - prior_p))
            # ).sum(dim=-1).mean()

            variational_loss = -var_logp_loss
            variational_loss.backward()

            var_loss_sum += variational_loss.item()
            var_slices += 1

            # -----------------------------
            # 4) MOTOR loss (unchanged, flat over all steps)
            # -----------------------------
            mini_batch_size = 16384
            total_steps = len(xy_pos_buf)

            for start in range(0, total_steps, mini_batch_size):
                end = min(start + mini_batch_size, total_steps)

                xy_slice = xy_pos[start:end]
                goal_slice = goalvec[start:end]
                chosen_slice = chosen_bandits_motor[start:end]

                mu, log_std = self.motor_fwd(
                    chosen_slice,
                    xy_slice,
                    goal_slice
                )

                dist = goal_slice.norm(dim=-1, keepdim=True) + 1e-6
                g_hat = goal_slice / dist
                speed = (dist / math.sqrt(8.0)).clamp(0.0, 1.0)
                target = (g_hat * speed).clamp(-0.999, 0.999)
                u_target = atanh(target)

                motor_loss_mini = F.mse_loss(mu, u_target)
                motor_loss_mini.backward()

                motor_loss_sum += float(motor_loss_mini.item())
                motor_slices += 1

            
            torch.nn.utils.clip_grad_norm_(list(self.bandit_parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(list(self.motor_parameters()), 1.0)
            optim_bandit.step()
            optim_motor.step()

        var_loss = var_loss_sum / max(1, var_slices)
        motor_loss = motor_loss_sum / max(1, motor_slices)
        return var_loss, motor_loss
