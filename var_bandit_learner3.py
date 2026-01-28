import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math


# ----------------------------
# Small helpers
# ----------------------------
def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def lr_repulsion_loss(left_feats, right_feats, margin=0.2):
    """Penalize left/right feature similarity so they don't collapse."""
    lf = left_feats.reshape(-1, left_feats.size(-1))
    rf = right_feats.reshape(-1, right_feats.size(-1))
    lf = F.normalize(lf, dim=-1)
    rf = F.normalize(rf, dim=-1)
    sim = (lf * rf).sum(dim=-1)          # cosine similarity
    return F.relu(sim - margin).mean()


def get_time_batch_indices_for_label(labels_tb: np.ndarray, target: str, device):
    """
    labels_tb: (T,B) numpy array of strings
    Returns:
      t_grid: (K,B) LongTensor
      b_grid: (K,B) LongTensor
      K: int
      B: int
    Assumes each batch has same number K of occurrences of target.
    """
    assert labels_tb.ndim == 2, f"Expected (T,B), got {labels_tb.shape}"
    T, B = labels_tb.shape

    times_per_b = [np.where(labels_tb[:, b] == target)[0] for b in range(B)]
    if max((len(x) for x in times_per_b), default=0) == 0:
        raise ValueError(f"No occurrences of target='{target}' found.")

    K = len(times_per_b[0])
    for b in range(B):
        if len(times_per_b[b]) != K:
            raise ValueError(
                f"Unequal counts for '{target}': batch0={K}, batch{b}={len(times_per_b[b])}."
            )

    t_grid_np = np.stack([np.sort(times_per_b[b]) for b in range(B)], axis=1)  # (K,B)
    b_grid_np = np.broadcast_to(np.arange(B)[None, :], (K, B)).copy()


    t_grid = torch.as_tensor(t_grid_np, device=device, dtype=torch.long)
    b_grid = torch.as_tensor(b_grid_np, device=device, dtype=torch.long)

    return t_grid, b_grid, K, B


def subset_TBH_grouped(tensor_tbh, t_grid, b_grid):
    """tensor_tbh: (T,B,...) and t_grid/b_grid: (K,B) => (K,B,...)"""
    return tensor_tbh[t_grid, b_grid]


# ----------------------------
# Vision encoder
# ----------------------------
class CNNEncoder(nn.Module):
    """A small CNN for processing visual inputs (expects NCHW)."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, feature_dim),
        )

    def forward(self, x):
        return self.cnn(x)


# ----------------------------
# Main learner
# ----------------------------
class BanditLearner(nn.Module):
    """
    HiT-DVAE-style sequential latent model with:
      - 3 encoder MHAs (self, teacher_AO, teacher_OL) using key_padding_mask (no manual memory filtering)
      - DVAE prior via z-space decoder MHA (dec_teacher) + prior_head
      - Action reconstruction from z using action_dec (+ optional reward recon for OL)
      - Actor-critic policy learning using carried teacher latent (AO-o -> AO-s, OL-o -> OL-s)
      - Motor policy unchanged
    """
    def __init__(self, input_size, feature_dim, rnn_hidden_size, action_dim,
                 z_dim=16, log_std_min=-5.0, log_std_max=-1.0):
        super().__init__()

        self.enc = CNNEncoder(feature_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.z_dim = z_dim

        # Pair embedding (left/right features)
        self.inp_emb = nn.Linear(2 * feature_dim, rnn_hidden_size, bias=False)

        # Token embeddings
        self.actor  = nn.Embedding(2, rnn_hidden_size)  # 0=self, 1=teacher
        self.action = nn.Embedding(2, rnn_hidden_size)  # 0=left, 1=right
        self.rwd_in = nn.Embedding(3, rnn_hidden_size)  # 0, 1, NO_FEEDBACK(2)

        # 3 encoder MHAs (rnn_hidden_size space)
        self.enc_self       = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, dropout=0.0)
        self.enc_teacher_ao = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, dropout=0.0)
        self.enc_teacher_ol = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, dropout=0.0)

        self.bos_token = nn.Parameter(torch.zeros(1, 1, rnn_hidden_size))


        # Stream norms
        self.self_ln       = nn.LayerNorm(rnn_hidden_size)
        self.teacher_ao_ln = nn.LayerNorm(rnn_hidden_size)
        self.teacher_ol_ln = nn.LayerNorm(rnn_hidden_size)

        # Heads for posterior q(z_t | h_t)
        self.post_head = nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size),
            nn.ReLU(),
            nn.Linear(rnn_hidden_size, 2 * z_dim),
        )

        # z-space decoder for prior p(z_{t+1}|...)
        self.dec_teacher = nn.MultiheadAttention(embed_dim=z_dim, num_heads=4, dropout=0.0)
        self.prior_head = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, 2 * z_dim),
        )

        # Project H->z for reconstruction conditioning
        self.H_to_z = nn.Linear(rnn_hidden_size, z_dim)

        # Reconstruction decoders (z-space query, z-space kv from shifted H_to_z(x))
        self.action_dec = nn.MultiheadAttention(embed_dim=z_dim, num_heads=4, dropout=0.0)
        self.reward_dec = nn.MultiheadAttention(embed_dim=z_dim, num_heads=4, dropout=0.0)
        self.a_logits = nn.Linear(z_dim, 2)
        self.r_logits = nn.Linear(z_dim, 1)

        self.attn_ln2 = nn.LayerNorm(z_dim)

        # Policy / critic (bandit choice)
        self.policy_net_il  = nn.Sequential(nn.Linear(rnn_hidden_size + z_dim, 128), nn.ReLU(), nn.Linear(128, 2))
        self.policy_net_obs = nn.Sequential(nn.Linear(rnn_hidden_size + 2 * z_dim, 128), nn.ReLU(), nn.Linear(128, 2))

        self.critic_net_il  = nn.Sequential(nn.Linear(rnn_hidden_size + z_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.critic_net_obs = nn.Sequential(nn.Linear(rnn_hidden_size + 2 * z_dim, 128), nn.ReLU(), nn.Linear(128, 1))

        # Motor policy (unchanged)
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

        # Optional: predicted reward probs for rollout policy (kept from your code)
        self.arm_reward_head = nn.Sequential(
            nn.Linear(rnn_hidden_size + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    # ----------------------------
    # Parameter groups
    # ----------------------------
    def bandit_parameters(self):
        modules = [
            self.enc,
            self.inp_emb,
            self.actor, self.action, self.rwd_in,
            self.enc_self, self.enc_teacher_ao, self.enc_teacher_ol,
            self.self_ln, self.teacher_ao_ln, self.teacher_ol_ln,
            self.post_head,
            self.dec_teacher, self.prior_head,
            self.H_to_z,
            self.action_dec, self.reward_dec,
            self.a_logits, self.r_logits,
            self.attn_ln2,
            self.policy_net_il, self.policy_net_obs,
            self.critic_net_il, self.critic_net_obs,
            self.arm_reward_head,
        ]
        for m in modules:
            for p in m.parameters():
                yield p

    def motor_parameters(self):
        modules = [self.mlp_pos, self.mlp_goal, self.mu_head, self.log_std_head]
        for m in modules:
            for p in m.parameters():
                yield p

    # ----------------------------
    # Forward helpers
    # ----------------------------
    def encode(self, obs_nchw):
        return self.enc(obs_nchw)

    def reward_compute(self, h, left_feats, right_feats):
        """
        h: (1,1,H) or (T,B,H) context
        left_feats/right_feats: (1,1,F) or (T,B,F)
        Returns p_left, p_right each (...,1)
        """
        left_in = torch.cat([h, left_feats], dim=-1)
        right_in = torch.cat([h, right_feats], dim=-1)
        left_p = torch.sigmoid(self.arm_reward_head(left_in))
        right_p = torch.sigmoid(self.arm_reward_head(right_in))
        return left_p, right_p

    def motor_fwd(self, choice_target, xy_pos=None, goal_vec=None):
        pos_emb  = self.mlp_pos(xy_pos)
        goal_emb = self.mlp_goal(goal_vec)
        # choice_target = choice_target.squeeze(1)
        # print("choice_target.shape=", choice_target.shape, ", pos_emb.shape=", pos_emb.shape, ", goal_emb.shape=", goal_emb.shape)
        motor_inp  = torch.cat([choice_target, pos_emb, goal_emb], dim=-1)
        mu      = self.mu_head(motor_inp)
        log_std = self.log_std_head(motor_inp).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def safe_attn(self, enc, x_all, attn_mask_T, kpm):
        # x_all: (T,B,D)
        T, B, D = x_all.shape

        # BOS token (must exist as a parameter)
        bos = self.bos_token.expand(1, B, D)     # (1,B,D)
        x2 = torch.cat([bos, x_all], dim=0)      # (T+1,B,D)

        # new causal mask for (T+1)
        attn_mask2 = torch.triu(
            torch.ones((T+1, T+1), device=x_all.device, dtype=torch.bool),
            diagonal=1
        )

        # prepend BOS to key padding mask: BOS is always valid => False
        bos_kpm = torch.zeros((kpm.size(0), 1), device=kpm.device, dtype=torch.bool)
        kpm2 = torch.cat([bos_kpm, kpm], dim=1)  # (B,T+1)

        out2, _ = enc(x2, x2, x2, attn_mask=attn_mask2, key_padding_mask=kpm2)

        return out2[1:]  # drop BOS output -> (T,B,D)

    def kl_normal(self, mu_q, log_std_q, mu_p, log_std_p):
        # log_std = log(sigma)
        var_q = torch.exp(2 * log_std_q)
        var_p = torch.exp(2 * log_std_p)
        return (log_std_p - log_std_q) + 0.5 * (var_q + (mu_q - mu_p).pow(2)) / var_p - 0.5




    # ----------------------------
    # Training update
    # ----------------------------
    def update2(self, optim_bandit, optim_motor, xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
                bandit_obs, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf,
                actor_buf, trial_cond_buf, device):
        """
        bandit_obs: list (B) of dicts with keys "left"/"right" each (T,C,H,W)
        chosen_bandits_buf: list (B) of (T,2) tensors
        bandit_rewards_buf: list (B) of (T,) tensors (0/1; AO-o may be 2, we will override anyway)
        actor_buf: list (B) of (T,) tensors (0=self,1=teacher)
        trial_cond_buf: list (B) of (T,) arrays/strings; will be made into (T,B) numpy
        """
        # -----------------------------
        # 1) MOTOR data (flat)
        # -----------------------------
        xy_pos = torch.as_tensor(np.stack(xy_pos_buf), device=device, dtype=torch.float32)
        goalvec = torch.as_tensor(np.asarray(goal_vec_buf, np.float32), device=device)
        chosen_bandits_motor = torch.as_tensor(np.stack(chosen_bandits_motor_buf), device=device, dtype=torch.float32)

        # -----------------------------
        # 2) BANDIT data (batched)
        # -----------------------------
        left_obs  = torch.stack([torch.tensor(b["left"],  device=device, dtype=torch.float32) for b in bandit_obs], dim=0)
        right_obs = torch.stack([torch.tensor(b["right"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)
        chosen_bandits = torch.stack(chosen_bandits_buf, dim=0).to(device)     # (B,T,2)
        rewards_bandits = torch.stack(bandit_rewards_buf, dim=0).to(device)    # (B,T)
        meta_ep_start = torch.stack(meta_ep_start_buf, dim=0).to(device)       # (B,T)
        actor = torch.stack(actor_buf, dim=0).to(device)                       # (B,T)
        trial_cond = np.array(trial_cond_buf)                                  # (B,T) strings

        B, T, C, H, W = left_obs.shape

        # time-major
        chosen_bandits = chosen_bandits.permute(1, 0, 2)   # (T,B,2)
        rewards_bandits = rewards_bandits.permute(1, 0)    # (T,B)
        meta_ep_start = meta_ep_start.permute(1, 0)        # (T,B)
        actor = actor.permute(1, 0)                        # (T,B)
        trial_cond = trial_cond.transpose(1, 0)            # (T,B) numpy strings

        rewards_rnn = rewards_bandits.unsqueeze(-1)        # (T,B,1)

        # indices for grouped subsequences (K,B)
        t_ao_o_idx, b_ao_o_idx, _, _ = get_time_batch_indices_for_label(trial_cond, "AO-o", device=device)
        t_ol_o_idx, b_ol_o_idx, _, _ = get_time_batch_indices_for_label(trial_cond, "OL-o", device=device)

        # final aligned buffers for policy loop
        z_final  = torch.zeros(T, B, self.z_dim, device=device)
        x_final  = torch.zeros(T, B, self.rnn_hidden_size, device=device)
        act_final = torch.zeros(T, B, device=device, dtype=torch.long)
        rew_final = torch.zeros(T, B, device=device)

        var_loss_sum, var_slices = 0.0, 0
        motor_loss_sum, motor_slices = 0.0, 0

        num_epochs = 1
        for _ in range(num_epochs):
            optim_motor.zero_grad(set_to_none=True)
            optim_bandit.zero_grad(set_to_none=True)

            # -----------------------------
            # 3) Encode left/right images -> features (T,B,F)
            # -----------------------------
            left_flat  = left_obs.reshape(B * T, C, H, W)
            right_flat = right_obs.reshape(B * T, C, H, W)

            left_feats  = self.encode(left_flat).view(B, T, -1).permute(1, 0, 2)
            right_feats = self.encode(right_flat).view(B, T, -1).permute(1, 0, 2)

            # -----------------------------
            # 4) Build aligned tokens (T,B,H)
            # -----------------------------
            action_all = torch.argmax(chosen_bandits, dim=-1)                         # (T,B)
            lr_concat_all = torch.cat([left_feats, right_feats], dim=-1)              # (T,B,2F)
            inp_all = self.inp_emb(lr_concat_all)                                     # (T,B,H)

            action_emb_all = self.action(action_all)
            actor_emb_all  = self.actor(actor)

            # reward indices: override AO-o to NO_FEEDBACK=2 regardless of buffer content
            reward_idx_all = rewards_rnn.squeeze(-1).long()
            mask_AOo = torch.as_tensor((trial_cond == "AO-o"), device=device)
            mask_OLo = torch.as_tensor((trial_cond == "OL-o"), device=device)
            reward_idx_all = torch.where(mask_AOo, torch.full_like(reward_idx_all, 2), reward_idx_all)
            rwd_emb_all = self.rwd_in(reward_idx_all)

            x_all = inp_all + action_emb_all + actor_emb_all + rwd_emb_all            # (T,B,H)
            x_dec_all = inp_all + actor_emb_all                                       # (T,B,H)

            # -----------------------------
            # 5) 3 encoder MHAs with key_padding_mask
            # -----------------------------
            mask_teacher_ao = mask_AOo
            mask_teacher_ol = mask_OLo
            mask_teacher_any = mask_teacher_ao | mask_teacher_ol
            mask_self = ~mask_teacher_any                                             # IL, AO-s, OL-s

            kpm_self = (~mask_self).T
            kpm_ao   = (~mask_teacher_ao).T
            kpm_ol   = (~mask_teacher_ol).T

            attn_mask_T = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)

            h_self = self.safe_attn(self.enc_self, x_all, attn_mask_T, kpm_self)
            h_self = self.self_ln(h_self + x_all)

            h_ao = self.safe_attn(self.enc_teacher_ao, x_all, attn_mask_T, kpm_ao)
            h_ao = self.teacher_ao_ln(h_ao + x_all)

            h_ol = self.safe_attn(self.enc_teacher_ol, x_all, attn_mask_T, kpm_ol)
            h_ol = self.teacher_ol_ln(h_ol + x_all)


            def sample_z(h):
                q = self.post_head(h)
                mu, log_std = q[..., :self.z_dim], q[..., self.z_dim:]
                log_std = log_std.clamp(-5.0, 2.0)
                eps = torch.randn_like(mu)
                z = mu + log_std.exp() * eps
                return z, mu, log_std

            z_self, mu_self, log_std_self = sample_z(h_self)
            z_ao,   mu_ao,   log_std_ao   = sample_z(h_ao)
            z_ol,   mu_ol,   log_std_ol   = sample_z(h_ol)


            # Fill aligned final buffers
            z_final[mask_self] = z_self[mask_self]
            x_final[mask_self] = x_dec_all[mask_self]
            act_final[mask_self] = action_all[mask_self]
            rew_final[mask_self] = rewards_rnn.squeeze(-1)[mask_self].float()

            z_final[mask_teacher_ao] = z_ao[mask_teacher_ao]
            x_final[mask_teacher_ao] = x_dec_all[mask_teacher_ao]
            act_final[mask_teacher_ao] = action_all[mask_teacher_ao]
            rew_final[mask_teacher_ao] = reward_idx_all[mask_teacher_ao].float()

            z_final[mask_teacher_ol] = z_ol[mask_teacher_ol]
            x_final[mask_teacher_ol] = x_dec_all[mask_teacher_ol]
            act_final[mask_teacher_ol] = action_all[mask_teacher_ol]
            rew_final[mask_teacher_ol] = rewards_rnn.squeeze(-1)[mask_teacher_ol].float()

            z_ac = torch.zeros(T, B, self.z_dim, device=device)
            z_ac[mask_self]       = mu_self[mask_self]
            z_ac[mask_teacher_ao] = mu_ao[mask_teacher_ao]
            z_ac[mask_teacher_ol] = mu_ol[mask_teacher_ol]


            # -----------------------------
            # 6) AO-o / OL-o DVAE losses on grouped subsequences (K,B)
            # -----------------------------
            # AO-o
            x_ao_o = subset_TBH_grouped(x_all, t_ao_o_idx, b_ao_o_idx)                 # (K,B,H)
            z_ao_o = subset_TBH_grouped(z_ao, t_ao_o_idx, b_ao_o_idx)                  # (K,B,z)
            mu_q_ao_o = subset_TBH_grouped(mu_ao, t_ao_o_idx, b_ao_o_idx)
            log_std_q_ao_o = subset_TBH_grouped(log_std_ao, t_ao_o_idx, b_ao_o_idx)
            action_ao_o = subset_TBH_grouped(action_all, t_ao_o_idx, b_ao_o_idx)

            K_ao = x_ao_o.size(0)
            Dz = self.z_dim

            # prior + KL
            if K_ao >= 2:
                attn_mask_Lm1 = torch.triu(torch.ones((K_ao - 1, K_ao - 1), device=device, dtype=torch.bool), diagonal=1)
                u_ao_tplus1, _ = self.dec_teacher(self.H_to_z(x_ao_o[:-1]), z_ao_o[:-1], z_ao_o[:-1], attn_mask=attn_mask_Lm1)
                u_ao_tplus1 = self.attn_ln2(u_ao_tplus1)

                p_params = self.prior_head(u_ao_tplus1)
                mu_prior, log_std_prior = p_params[..., :Dz], p_params[..., Dz:]
                log_std_prior = log_std_prior.clamp(-5.0, 2.0)

                mu_q_next = mu_q_ao_o[1:]
                log_std_q_next = log_std_q_ao_o[1:]
                # kl = (log_std_q_next - log_std_prior +
                #       (log_std_q_next.exp().pow(2) + (mu_q_next - mu_prior).pow(2)) /
                #       (2.0 * log_std_prior.exp().pow(2)) - 0.5)
                # kl_err_ao = kl.sum(-1).mean()
                kl = self.kl_normal(mu_q_next, log_std_q_next, mu_prior, log_std_prior)
                kl_err_ao = kl.sum(-1).mean()

            else:
                kl_err_ao = torch.zeros((), device=device)

            # action reconstruction
            x_ao_o_z = self.H_to_z(x_ao_o)
            bos = torch.zeros(1, B, Dz, device=device)
            kv = torch.cat([bos, x_ao_o_z[:-1]], dim=0)
            attn_mask_rec = torch.triu(torch.ones((K_ao, K_ao), device=device, dtype=torch.bool), diagonal=1)
            out_a, _ = self.action_dec(z_ao_o, kv, kv, attn_mask=attn_mask_rec)
            logits_a = self.a_logits(out_a)
            ao_choice_loss = F.cross_entropy(logits_a.reshape(-1, 2), action_ao_o.reshape(-1))
            ao_reward_loss = torch.zeros((), device=device)  # no feedback in AO-o

            # OL-o
            x_ol_o = subset_TBH_grouped(x_all, t_ol_o_idx, b_ol_o_idx)
            z_ol_o = subset_TBH_grouped(z_ol, t_ol_o_idx, b_ol_o_idx)
            mu_q_ol_o = subset_TBH_grouped(mu_ol, t_ol_o_idx, b_ol_o_idx)
            log_std_q_ol_o = subset_TBH_grouped(log_std_ol, t_ol_o_idx, b_ol_o_idx)
            action_ol_o = subset_TBH_grouped(action_all, t_ol_o_idx, b_ol_o_idx)
            rewards_ol_o = subset_TBH_grouped(rewards_rnn, t_ol_o_idx, b_ol_o_idx)

            K_ol = x_ol_o.size(0)
            if K_ol >= 2:
                attn_mask_Lm1 = torch.triu(torch.ones((K_ol - 1, K_ol - 1), device=device, dtype=torch.bool), diagonal=1)
                u_ol_tplus1, _ = self.dec_teacher(self.H_to_z(x_ol_o[:-1]), z_ol_o[:-1], z_ol_o[:-1], attn_mask=attn_mask_Lm1)
                u_ol_tplus1 = self.attn_ln2(u_ol_tplus1)

                p_params = self.prior_head(u_ol_tplus1)
                mu_prior, log_std_prior = p_params[..., :Dz], p_params[..., Dz:]
                log_std_prior = log_std_prior.clamp(-5.0, 2.0)

                mu_q_next = mu_q_ol_o[1:]
                log_std_q_next = log_std_q_ol_o[1:]
                # kl = (log_std_q_next - log_std_prior +
                #       (log_std_q_next.exp().pow(2) + (mu_q_next - mu_prior).pow(2)) /
                #       (2.0 * log_std_prior.exp().pow(2)) - 0.5)
                kl = self.kl_normal(mu_q_next, log_std_q_next, mu_prior, log_std_prior)
                kl_err_ol = kl.sum(-1).mean()
            else:
                kl_err_ol = torch.zeros((), device=device)

            x_ol_o_z = self.H_to_z(x_ol_o)
            bos = torch.zeros(1, B, Dz, device=device)
            kv = torch.cat([bos, x_ol_o_z[:-1]], dim=0)
            attn_mask_rec = torch.triu(torch.ones((K_ol, K_ol), device=device, dtype=torch.bool), diagonal=1)

            out_a, _ = self.action_dec(z_ol_o, kv, kv, attn_mask=attn_mask_rec)
            logits_a = self.a_logits(out_a)
            ol_choice_loss = F.cross_entropy(logits_a.reshape(-1, 2), action_ol_o.reshape(-1))

            out_r, _ = self.reward_dec(z_ol_o, kv, kv, attn_mask=attn_mask_rec)
            logits_r = self.r_logits(out_r)
            ol_reward_loss = F.binary_cross_entropy_with_logits(logits_r, rewards_ol_o.float())

            # -----------------------------
            # 7) Actor-critic loss over aligned timeline
            # -----------------------------
            actor_loss = torch.zeros((), device=device)
            critic_loss = torch.zeros((), device=device)
            count = 0

            advs = []
            logpis = []
            entropies = []
            values = []
            rewards = []


            for b in range(B):
                z_ao_o_last = torch.zeros(self.z_dim, device=device)
                z_ol_o_last = torch.zeros(self.z_dim, device=device)
                z_il_tm1 = torch.zeros(self.z_dim, device=device)
                z_ao_s_tm1 = torch.zeros(self.z_dim, device=device)
                z_ol_s_tm1 = torch.zeros(self.z_dim, device=device)

                for t in range(T):
                    lab = trial_cond[t, b]
                    if lab == "AO-o":
                        # z_ao_o_last = z_final[t, b]
                        z_ao_o_last = z_ac[t,b]
                        continue
                    if lab == "OL-o":
                        # z_ol_o_last = z_final[t, b]
                        z_ol_o_last = z_ac[t,b]
                        continue

                    if lab == "IL":
                        pol_logits = self.policy_net_il(torch.cat([x_final[t, b], z_il_tm1], dim=-1))
                        val = self.critic_net_il(torch.cat([x_final[t, b], z_il_tm1], dim=-1)).squeeze(-1)
                        # z_il_tm1 = z_final[t, b]
                        z_il_tm1 = z_ac[t,b]
                    elif lab == "AO-s":
                        pol_logits = self.policy_net_obs(torch.cat([x_final[t, b], z_ao_s_tm1, z_ao_o_last], dim=-1))
                        val = self.critic_net_obs(torch.cat([x_final[t, b], z_ao_s_tm1, z_ao_o_last], dim=-1)).squeeze(-1)
                        # z_ao_s_tm1 = z_final[t, b]
                        z_ao_s_tm1 = z_ac[t,b]
                    elif lab == "OL-s":
                        pol_logits = self.policy_net_obs(torch.cat([x_final[t, b], z_ol_s_tm1, z_ol_o_last], dim=-1))
                        val = self.critic_net_obs(torch.cat([x_final[t, b], z_ol_s_tm1, z_ol_o_last], dim=-1)).squeeze(-1)
                        # z_ol_s_tm1 = z_final[t, b]
                        z_ol_s_tm1 = z_ac[t,b]
                    else:
                        continue

                    # pol_logits = torch.nan_to_num(pol_logits, nan=0.0, posinf=0.0, neginf=0.0)
                    # pol_logits = pol_logits.clamp(-20, 20)
                    tau = 0.2
                    dist = torch.distributions.Categorical(logits=pol_logits/tau)
                    logpi = dist.log_prob(act_final[t, b])
                    # val = rew_final.mean()
                    adv = (rew_final[t, b] - val).detach()
                    ent = dist.entropy()

                    # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # actor_loss += (-(adv * logpi))
                    # critic_loss += (rew_final[t, b] - val).pow(2)
                    advs.append(adv)
                    logpis.append(logpi)
                    entropies.append(ent)
                    values.append(val)
                    rewards.append(rew_final[t, b])


            advs = torch.stack(advs)               # [N]
            logpis = torch.stack(logpis)           # [N]
            entropies = torch.stack(entropies)     # [N]

            adv_mean = advs.mean()
            adv_std  = advs.std(unbiased=False) + 1e-8
            advs_n = (advs - adv_mean) / adv_std

            # actor loss with normalized advantage
            entropy_coef = 0.01
            actor_loss = -(advs_n * logpis).mean() - entropy_coef * entropies.mean()

            # critic loss (MSE)
            values = torch.stack(values)
            rewards = torch.stack(rewards)
            critic_loss = 0.5 * (rewards - values).pow(2).mean()


            # -----------------------------
            # 8) Total bandit loss
            # -----------------------------
            contrast = lr_repulsion_loss(left_feats, right_feats, margin=0.2)
            lambda_contrast = 0.05
            w_ac = 1.0
            w_recon = 1.0
            w_kl = 0.05


            beh_loss = (w_ac * (actor_loss + critic_loss) +
            w_recon * (ao_choice_loss + ol_choice_loss + ol_reward_loss) +
            w_kl * (kl_err_ao + kl_err_ol))

            print("actor_loss =",round(actor_loss.item(),2), ", critic_loss =", round(critic_loss.item(),2), "ao_choice_loss =", round(ao_choice_loss.item(),2), ", ol_choice_loss =", round(ol_choice_loss.item(),2), ", kl_err_ao =", round(kl_err_ao.item(),2), ", kl_err_ol=", round(kl_err_ol.item(),2))
            bandit_total = beh_loss + lambda_contrast * contrast
            bandit_total.backward()

            var_loss_sum += float(beh_loss.item())
            var_slices += 1

            # -----------------------------
            # 9) Motor loss (unchanged)
            # -----------------------------
            mini_batch_size = 16384
            total_steps = len(xy_pos_buf)

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
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if not torch.isfinite(p).all():
                        print("NON-FINITE WEIGHT AFTER STEP:", n)
                        return float("nan"), float("nan")




        var_loss = var_loss_sum / max(1, var_slices)
        motor_loss = motor_loss_sum / max(1, motor_slices)
        return var_loss, motor_loss
