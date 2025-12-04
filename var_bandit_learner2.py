# Modified var_bandit_learner2.py
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
import math

class CNNEncoder(nn.Module):
    """
    Pretrained CNN feature extractor -> projects to feature_dim.
    Expects NCHW images in [0,1] (like your current code).
    """
    def __init__(self, feature_dim: int, train_backbone: bool = False, normalize: bool = False):
        super().__init__()
        self.normalize = normalize

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        layers = list(backbone.children())[:-1]  # remove FC
        self.backbone = nn.Sequential(*layers)

        # Freeze backbone if not training
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ResNet18 outputs 512
        self.proj = nn.Linear(512, feature_dim)

        # ImageNet normalization buffers
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) range [0,1]
        if self.normalize:
            x = (x - self.mean) / (self.std + 1e-8)
        feat = self.backbone(x)          # (B,512,1,1)
        feat = feat.view(x.size(0), -1)  # (B,512)
        return self.proj(feat)           # (B,feature_dim)


def gaussian_log_prob(actions, mean, log_std):
    """
    Gaussian log prob for a tanh-squashed Gaussian policy.
    actions, mean: (B,A)
    log_std: (B,A)
    """
    std = torch.exp(log_std)
    var = std.pow(2)
    # pre-tanh action is not available here, but the code uses correction separately (see below).
    # This gaussian_log_prob is used with a pretanh action provided elsewhere.
    log_scale = log_std
    log_prob = -((actions - mean) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    return log_prob.sum(dim=-1)  # (B,)


def gaussian_log_prob_with_tanh_correction(actions, mean, log_std, pretahn_actions):
    """
    Returns log pi(a) for tanh-squashed Gaussian with correction.
    pretahn_actions: the action before tanh, same shape as actions (B,A)
    """
    std = torch.exp(log_std)
    var = std.pow(2)
    log_scale = log_std
    log_prob = -((pretahn_actions - mean) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))
    log_prob = log_prob.sum(dim=-1)  # sum over action dimensions

    corr = torch.log(1 - torch.tanh(pretahn_actions).pow(2) + 1e-6).sum(dim=-1)  # correction term
    return log_prob - corr  # (B,)


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class ShuffleNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = shufflenet_v2_x0_5(weights=weights)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.maxpool,
            backbone.stage2,
            backbone.stage3,
            backbone.stage4
        )

        self.out_dim = 176  # final feature size
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        return x.flatten(1)   # (B,176)



class BanditLearner(nn.Module):
    def __init__(self, input_size, feature_dim, rnn_hidden_size, action_dim,
                 log_std_min=-5.0, log_std_max=-1.0,
                 num_pairs=None, max_trials=None):
        nn.Module.__init__(self)

        self.input_size = input_size

        self.downsample = nn.AvgPool2d(kernel_size=4, stride=4)

        self.debug_gru_inputs = {"rollout": [], "update": []}

        # self.enc = CNNEncoder(feature_dim, train_backbone=False)
        self.enc = ShuffleNetEncoder(pretrained=True)

        # self.enc = CNNEncoder(feature_dim)

        # --- Bandit RNN + heads replaced with Transformer
        self.hidden_size = rnn_hidden_size
        self.input_proj = nn.Linear(input_size, self.hidden_size)
        self.pos_enc = PositionalEncoding(self.hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=4, dim_feedforward=256, dropout=0.0),
            num_layers=2
        )

        self.rewards_head = nn.Sequential(
            nn.Linear(self.hidden_size + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        # --- Motor net
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

    def generate_square_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def transformer_fwd_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Run the causal Transformer over a token sequence.

        tokens:
            (S, B, D) or (S, D)

        returns:
            (S, B, H) or (S, H)
        """
        batched = (tokens.dim() == 3)
        if not batched:
            tokens = tokens.unsqueeze(1)  # (S, 1, D)

        seq = self.input_proj(tokens)  # (S, B, H)
        seq = self.pos_enc(seq)
        S = seq.size(0)
        mask = self.generate_square_subsequent_mask(S, seq.device)
        out = self.transformer(seq, mask=mask)  # (S, B, H)

        if not batched:
            out = out.squeeze(1)  # (S, H)
        return out

    def rnn_fwd(self, features, action, reward, meta_ep_start, h=None, history=None):
        x = torch.cat([features, action, reward, meta_ep_start], dim=-1).to(features.dtype).to(features.device)

        if history is not None:
            # x should be (1, D). Keep it 2D so it matches history (t, D).
            if x.dim() == 1:
                x = x.unsqueeze(0)  # safety

            history = torch.cat([history, x], dim=0)  # (t+1, D)

            seq_len = history.size(0)
            seq = history.unsqueeze(1)  # (seq_len, 1, D)
            seq = self.input_proj(seq)
            seq = self.pos_enc(seq)
            mask = self.generate_square_subsequent_mask(seq_len, seq.device)
            out = self.transformer(seq, mask=mask)  # (seq_len, 1, hidden)
            new_h = out[-1]  # (1, hidden)
            return out[-1].squeeze(0), new_h, history

        else:
            # batched full sequence mode
            batched = x.dim() == 3
            if not batched:
                x = x.unsqueeze(1)
            seq = self.input_proj(x)
            seq = self.pos_enc(seq)
            seq_len = seq.size(0)
            mask = self.generate_square_subsequent_mask(seq_len, seq.device)
            out = self.transformer(seq, mask=mask)  # (T, B, H)
            new_h = out[-1].unsqueeze(0)  # (1, B, H)
            if not batched:
                out = out.squeeze(1)
                new_h = new_h.squeeze(1)
            return out, new_h

    ...
    def reward_compute(self, rnn_out, curr_feat):
        # rnn_out: (S,H) or (1,H) or (T,B,H)
        # curr_feat: (S,F) or (1,F) or (T,B,F)
        x = torch.cat([rnn_out, curr_feat], dim=-1)
        return self.rewards_head(x)  # (...,2)

    def motor_fwd(self, choice_target, xy_pos=None, goal_vec=None):
        pos_emb = self.mlp_pos(xy_pos)       # (T,32) or (B,32)
        goal_emb = self.mlp_goal(goal_vec)   # (T,32) or (B,32)
        motor_inp = torch.cat([choice_target, pos_emb, goal_emb], dim=-1)
        mu = self.mu_head(motor_inp)
        log_std = self.log_std_head(motor_inp).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def bandit_parameters(self):
        modules = []
        modules += [
            self.transformer,
            self.input_proj,
            self.pos_enc,
            self.rewards_head,
            self.enc,
        ]
        for m in modules:
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
        goal_vec = torch.as_tensor(np.stack(goal_vec_buf), device=device, dtype=torch.float32)
        chosen_bandits_motor = torch.as_tensor(np.stack(chosen_bandits_motor_buf), device=device, dtype=torch.float32)

        # -----------------------------
        # 2) BANDIT data: stack into batch
        # -----------------------------
        bandit_obs = torch.as_tensor(np.stack(bandit_obs), device=device, dtype=torch.float32)
        chosen_bandits = torch.as_tensor(np.stack(chosen_bandits_buf), device=device, dtype=torch.float32)
        rewards_bandits = torch.as_tensor(np.stack(bandit_rewards_buf), device=device, dtype=torch.float32)
        meta_ep_start = torch.as_tensor(np.stack(meta_ep_start_buf), device=device, dtype=torch.float32)

        # bandit_obs is already features: (B, T, F)
        B, T, Fdim = bandit_obs.shape

        # C, H, W = bandit_obs.shape[2:]

        # Reorder to time-major (T, B, ...)
        bandit_obs = bandit_obs.permute(1, 0, 2)         # (T, B, F)
        feats_bandit = bandit_obs
        chosen_bandits = chosen_bandits.permute(1, 0, 2)           # (T, B, 2)
        rewards_bandits = rewards_bandits.permute(1, 0)            # (T, B)
        meta_ep_start   = meta_ep_start.permute(1, 0)              # (T, B)

        # Encode CNN features: flatten (T, B) -> (T*B, C, H, W)
        # bandit_obs_flat = bandit_obs.reshape(T * B, C, H, W)       # (T*B, C, H, W)
                                 # (T, B, F)

        rewards_rnn = rewards_bandits.unsqueeze(-1)    # (T, B, 1)
        start_rnn   = meta_ep_start.unsqueeze(-1)      # (T, B, 1)

        num_epochs = 1

        var_loss_sum, var_slices = 0.0, 0
        motor_loss_sum, motor_slices = 0.0, 0

        for _ in range(num_epochs):

            optim_motor.zero_grad(set_to_none=True)
            optim_bandit.zero_grad(set_to_none=True)

            # -----------------------------
            # 3) BANDIT loss (Query token attention)
            # -----------------------------
            # feats_flat = self.encode(bandit_obs_flat)         # (T*B, F)
            # feats_bandit = feats_flat.view(T, B, -1)          # (T, B, F)

            # Build interleaved [Q0,M0,Q1,M1,...] tokens so the query token attends to past,
            # but cannot see the current trial's (action,reward) memory token (causal mask).
            zeros_a = torch.zeros_like(chosen_bandits)   # (T,B,2)
            zeros_r = torch.zeros_like(rewards_rnn)      # (T,B,1)

            # Query token at trial start: context only (no action/reward)
            xQ = torch.cat([feats_bandit, zeros_a, zeros_r, start_rnn], dim=-1)                 # (T,B,D)

            # Memory token at trial end: context + chosen action + obtained reward
            xM = torch.cat([feats_bandit, chosen_bandits, rewards_rnn, start_rnn], dim=-1)     # (T,B,D)

            # Interleave Q and M: Q0,M0,Q1,M1,...
            x = torch.empty((2 * T, B, self.input_size), device=device, dtype=feats_bandit.dtype)
            x[0::2] = xQ
            x[1::2] = xM

            out = self.transformer_fwd_tokens(x)   # (2T,B,H)
            q_out = out[0::2]                      # (T,B,H)

            reward_logits = self.reward_compute(q_out, feats_bandit)  # (T,B,2)
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
            ).mean()

            # variational_loss = -(var_logp_loss - 0.1 * var_kl_loss)
            variational_loss = -var_logp_loss

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
                goal_slice = goal_vec[start:end]
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

            variational_loss.backward()

            var_loss_sum += variational_loss.item()
            var_slices += 1

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
