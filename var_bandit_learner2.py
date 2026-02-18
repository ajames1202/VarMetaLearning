import math
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Small CNN for visual inputs (NCHW). Uses AdaptiveAvgPool2d to be resolution-agnostic."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


def atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x + 1e-6) - torch.log1p(-x + 1e-6))


def log_prob_tanh_normal(
    action: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    std = torch.exp(log_std)
    pre = torch.atanh(action.clamp(-1 + eps, 1 - eps))
    normal = torch.distributions.Normal(mean, std)
    logp = normal.log_prob(pre).sum(dim=-1)
    corr = torch.log(1 - torch.tanh(pre).pow(2) + eps).sum(dim=-1)
    return logp - corr


class BanditLearner(nn.Module):
    """Sequential DVAE with two distinct latents and joint low-rank cross-correlation.

    Design goals (per your request):
      - z_t = [z_self_t, z_obs_t]
      - Decode *self* reward/policy from z_self
      - Decode teacher behavior (action + optional teacher reward when available) from z_obs
      - Keep *joint* (low-rank) covariance so teacher evidence can influence z_self
      - Remove extra hand-designed posterior blending/gating; posterior is computed directly
        via dedicated post-heads for IL/AO/OL.

    Notes:
      - This is a training-time inference model (uses q(z|h,x)). For rollout/action selection,
        you typically sample from the prior p(z|h) before observing reward.
    """

    def __init__(
        self,
        input_size: int,
        feature_dim: int,
        rnn_hidden_size: int,
        action_dim: int,
        log_std_min: float = -5.0,
        log_std_max: float = -1.0,
        num_pairs=None,
        max_trials=None,
    ):
        super().__init__()

        self.enc = CNNEncoder(feature_dim)

        # Causal attention memory
        self.attn = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, dropout=0.1, batch_first=False)
        self.attn_ln = nn.LayerNorm(rnn_hidden_size)
        self.inp_emb = nn.Linear(2 * feature_dim, rnn_hidden_size, bias=False)

        # BOS token prevents NaNs when key_padding_mask masks all keys
        self.bos_token = nn.Parameter(torch.zeros(1, 1, rnn_hidden_size))

        # Token feature embeddings
        self.actor = nn.Embedding(2, rnn_hidden_size)   # 0=self, 1=teacher
        self.action = nn.Embedding(2, rnn_hidden_size)  # 0=left, 1=right
        self.rwd_in = nn.Embedding(3, rnn_hidden_size)  # 0,1,2(no_feedback)

        # -------------------------
        # Two-latent sequential DVAE (joint low-rank Gaussian)
        # -------------------------
        self.z_self_dim = 6
        self.z_obs_dim = 6
        self.z_dim = self.z_self_dim + self.z_obs_dim
        self.z_rank = 3  # set to 0 for diagonal Normal + analytic KL

        D, r = self.z_dim, self.z_rank
        self._z_out = (2 * D) + (D * r if r > 0 else 0)

        # Prior p(z_t | h_{t-1}) over the *joint* latent
        self.prior_net = nn.Sequential(
            nn.Linear(rnn_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self._z_out),
        )

        # Condition-specific amortized posteriors q(z_t | h_{t-1}, x_t, cond)
        # Dedicated heads for IL / AO / OL as requested.
        self.post_IL = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self._z_out),
        )
        self.post_AO = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self._z_out),
        )
        self.post_OL = nn.Sequential(
            nn.Linear(2 * rnn_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self._z_out),
        )

        # -------------------------
        # Decoders
        # -------------------------
        # Self reward decoder: p(r_self_t | z_self_t, h_{t-1}, s_t, a_t)
        self.self_reward_dec = nn.Sequential(
            nn.Linear(self.z_self_dim + rnn_hidden_size + 2 * feature_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Self policy decoder: p(a_self_t | z_self_t, h_{t-1}, s_t)
        self.self_policy_dec = nn.Sequential(
            nn.Linear(self.z_self_dim + rnn_hidden_size + 2 * feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        # Teacher behavior decoder: p(a_teacher_t | z_obs_t, h_{t-1}, s_t)
        self.teacher_action_dec = nn.Sequential(
            nn.Linear(self.z_obs_dim + rnn_hidden_size + 2 * feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        # Optional: teacher reward decoder when teacher feedback is observed (e.g., OL-o)
        self.teacher_reward_dec = nn.Sequential(
            nn.Linear(self.z_obs_dim + rnn_hidden_size + 2 * feature_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # -------------------------
        # Motor policy (unchanged)
        # -------------------------
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.mlp_pos = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.mlp_goal = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        motor_in = action_dim + 32 + 32
        self.mu_head = nn.Sequential(
            nn.Linear(motor_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(motor_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    # -------------------------
    # Parameter groups
    # -------------------------
    def bandit_parameters(self):
        modules = [
            self.enc,
            self.attn,
            self.attn_ln,
            self.inp_emb,
            self.actor,
            self.action,
            self.rwd_in,
            self.prior_net,
            self.post_IL,
            self.post_AO,
            self.post_OL,
            self.self_reward_dec,
            self.self_policy_dec,
            self.teacher_action_dec,
            self.teacher_reward_dec,
        ]
        for m in modules:
            for p in m.parameters():
                yield p
        yield self.bos_token

    def motor_parameters(self):
        modules = [self.mlp_pos, self.mlp_goal, self.mu_head, self.log_std_head]
        for m in modules:
            for p in m.parameters():
                yield p

    # -------------------------
    # Helpers
    # -------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def motor_fwd(self, choice_target: torch.Tensor, xy_pos: torch.Tensor, goal_vec: torch.Tensor):
        pos_emb = self.mlp_pos(xy_pos)
        goal_emb = self.mlp_goal(goal_vec)
        motor_inp = torch.cat([choice_target, pos_emb, goal_emb], dim=-1)
        mu = self.mu_head(motor_inp)
        log_std = self.log_std_head(motor_inp).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def _safe_causal_attn(
        self,
        query: torch.Tensor,  # (L,B,H)
        key: torch.Tensor,    # (L,B,H)
        value: torch.Tensor,  # (L,B,H)
        key_padding_mask: torch.Tensor,  # (B,L) True=ignore
    ) -> torch.Tensor:
        """Causal attention with a BOS token so we never softmax over all-masked keys."""
        Lq, B, H = query.shape
        dev, dtype = query.device, query.dtype

        bos_k = self.bos_token.expand(1, B, H)
        bos_v = torch.zeros((1, B, H), device=dev, dtype=dtype)
        key2 = torch.cat([bos_k, key], dim=0)   # (L+1,B,H)
        val2 = torch.cat([bos_v, value], dim=0)

        bos_kpm = torch.zeros((B, 1), device=dev, dtype=torch.bool)
        kpm2 = torch.cat([bos_kpm, key_padding_mask], dim=1)  # (B,L+1)

        key_len = key.shape[0]
        src_len = key_len + 1
        offset = max(0, key_len - Lq)
        attn_mask = torch.triu(
            torch.ones((Lq, src_len), device=dev, dtype=torch.bool),
            diagonal=offset + 2,
        )

        out, _ = self.attn(
            query=query,
            key=key2,
            value=val2,
            attn_mask=attn_mask,
            key_padding_mask=kpm2,
            need_weights=False,
        )
        return out

    def _rep_all(self, x_tok: torch.Tensor) -> torch.Tensor:
        """rep[t] is a causal representation that depends on x_{<=t}.

        The DVAE prior uses h_prev[t] = rep[t-1] so p(z_t|h_{t-1}) never sees x_t.
        """
        T, B, H = x_tok.shape
        if T == 1:
            return self.attn_ln(x_tok)

        q = x_tok[1:]     # (T-1,B,H)
        k = x_tok[:-1]
        v = x_tok[:-1]
        kpm = torch.zeros((B, T - 1), device=x_tok.device, dtype=torch.bool)
        attn_out = self._safe_causal_attn(q, k, v, kpm)

        rep = x_tok.new_zeros(T, B, H)
        rep[0] = self.attn_ln(x_tok[0])
        rep[1:] = self.attn_ln(attn_out + q)
        return rep

    def _unpack_joint_lr(self, out: torch.Tensor, T: int, B: int):
        """Unpack (mu, diag_raw, U) from a joint output."""
        D, r = self.z_dim, self.z_rank
        mu = out[..., :D]
        diag_raw = out[..., D: 2 * D]
        if r > 0:
            U = out[..., 2 * D :].view(T, B, D, r)
        else:
            U = None
        return mu, diag_raw, U

    # -------------------------
    # Main update
    # -------------------------
    def update2(
        self,
        optim_bandit,
        optim_motor,
        xy_pos_buf,
        goal_vec_buf,
        chosen_bandits_motor_buf,
        bandit_obs: List[Dict[str, Any]],
        chosen_bandits_buf: List[torch.Tensor],
        bandit_rewards_buf: List[torch.Tensor],
        meta_ep_start_buf: List[torch.Tensor],
        actor_buf: List[torch.Tensor],
        trial_cond_buf,
        device,
    ):
        # -----------------------------
        # 1) MOTOR data: keep flat
        # -----------------------------
        xy_pos = torch.as_tensor(np.stack(xy_pos_buf), device=device, dtype=torch.float32)
        goalvec = torch.as_tensor(np.asarray(goal_vec_buf, np.float32), device=device)
        chosen_bandits_motor = torch.as_tensor(np.stack(chosen_bandits_motor_buf), device=device, dtype=torch.float32)

        # -----------------------------
        # 2) BANDIT data: batched (B,T,...)
        # -----------------------------
        left_obs = torch.stack([torch.as_tensor(b["left"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)
        right_obs = torch.stack([torch.as_tensor(b["right"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)
        chosen_bandits = torch.stack(chosen_bandits_buf, dim=0).to(device)      # (B,T,2)
        rewards_bandits = torch.stack(bandit_rewards_buf, dim=0).to(device)     # (B,T)
        actor = torch.stack(actor_buf, dim=0).to(device)                        # (B,T)
        trial_cond = np.stack(trial_cond_buf, axis=0)                           # (B,T) strings

        B, T, C, H, W = left_obs.shape

        # time-major
        chosen_bandits = chosen_bandits.permute(1, 0, 2)   # (T,B,2)
        rewards_bandits = rewards_bandits.permute(1, 0)    # (T,B)
        actor = actor.permute(1, 0).long().clamp(0, 1)      # (T,B)
        trial_cond = trial_cond.transpose(1, 0)             # (T,B)

        # Reward index: 0/1 observed, 2 is NO_FEEDBACK
        rwd_idx = rewards_bandits.long().clamp(0, 2)
        rwd_obs = (rwd_idx != 2)
        rwd01 = rwd_idx.float().clamp(0.0, 1.0)

        num_epochs = 1
        bandit_loss_sum, bandit_slices = 0.0, 0
        motor_loss_sum, motor_slices = 0.0, 0

        for _ in range(num_epochs):
            optim_motor.zero_grad(set_to_none=True)
            optim_bandit.zero_grad(set_to_none=True)

            # -----------------------------
            # 3) BANDIT loss
            # -----------------------------
            left_flat = left_obs.reshape(B * T, C, H, W)
            right_flat = right_obs.reshape(B * T, C, H, W)

            left_feats = self.encode(left_flat).view(B, T, -1).permute(1, 0, 2)    # (T,B,F)
            right_feats = self.encode(right_flat).view(B, T, -1).permute(1, 0, 2)  # (T,B,F)
            lr_concat = torch.cat([left_feats, right_feats], dim=-1)               # (T,B,2F)

            stim_tok = self.inp_emb(lr_concat)                                     # (T,B,H)

            # action indices / embeddings
            a_t = torch.argmax(chosen_bandits, dim=-1)  # (T,B)
            action_emb = self.action(a_t)
            actor_emb = self.actor(actor)
            rwd_emb = self.rwd_in(rwd_idx)
            

            # x_t for the posterior can include reward (observed or NO_FEEDBACK)
            x_tok = stim_tok + actor_emb + action_emb + rwd_emb

            if T < 1:
                bandit_total = torch.zeros((), device=device)
            else:
                # -----------------------------
                # Condition masks (T,B) -> choose posterior head
                # -----------------------------
                mask_IL = torch.as_tensor(trial_cond == "IL", device=device)
                mask_AOo = torch.as_tensor(trial_cond == "AO-o", device=device)
                mask_AOs = torch.as_tensor(trial_cond == "AO-s", device=device)
                mask_OLo = torch.as_tensor(trial_cond == "OL-o", device=device)
                mask_OLs = torch.as_tensor(trial_cond == "OL-s", device=device)

                mask_AO = mask_AOo | mask_AOs
                mask_OL = mask_OLo | mask_OLs

                # -----------------------------
                # Causal representation rep[t] over full history
                # -----------------------------
                rep = self._rep_all(x_tok)   # (T,B,H)
                h_prev = rep.new_zeros(rep.shape)
                if T > 1:
                    h_prev[1:] = rep[:-1]

                # -----------------------------
                # DVAE: joint prior / posterior (low-rank)
                # -----------------------------
                D, r = self.z_dim, self.z_rank
                eps = 1e-4

                prior_out = self.prior_net(h_prev)  # (T,B,*)
                mu_p, diag_p_raw, U_p = self._unpack_joint_lr(prior_out, T, B)

                post_in = torch.cat([h_prev, x_tok], dim=-1)
                out_il = self.post_IL(post_in)
                out_ao = self.post_AO(post_in)
                out_ol = self.post_OL(post_in)

                # Select head per timestep
                w_il = mask_IL.float().unsqueeze(-1)
                w_ao = mask_AO.float().unsqueeze(-1)
                w_ol = mask_OL.float().unsqueeze(-1)
                out_q = out_il * w_il + out_ao * w_ao + out_ol * w_ol

                mu_q, diag_q_raw, U_q = self._unpack_joint_lr(out_q, T, B)

                if r > 0:
                    cov_diag_p = F.softplus(diag_p_raw) + eps
                    cov_diag_q = F.softplus(diag_q_raw) + eps

                    p_dist = torch.distributions.LowRankMultivariateNormal(mu_p, U_p, cov_diag_p)
                    q_dist = torch.distributions.LowRankMultivariateNormal(mu_q, U_q, cov_diag_q)

                    z = q_dist.rsample()  # (T,B,D)
                    z_p = p_dist.rsample()  # (T,B,D)
                    kl_tb = q_dist.log_prob(z) - p_dist.log_prob(z)  # (T,B) MC KL
                else:
                    sig_p = F.softplus(diag_p_raw) + eps
                    sig_q = F.softplus(diag_q_raw) + eps

                    q_dist = torch.distributions.Normal(mu_q, sig_q)
                    z = q_dist.rsample()
                    z_p = p_dist.rsample()  # (T,B,D)

                    kl_dim = torch.log(sig_p / sig_q) + (sig_q.pow(2) + (mu_q - mu_p).pow(2)) / (2.0 * sig_p.pow(2)) - 0.5
                    kl_tb = kl_dim.sum(dim=-1)

                kl_loss = kl_tb.mean()

                # Split latents
                z_self = z[..., : self.z_self_dim]
                z_obs = z[..., self.z_self_dim :]

                # -----------------------------
                # Decode: self reward + self policy from z_self
                # -----------------------------
                self_evt = (actor == 0)
                teacher_evt = (actor == 1)

                # # self policy loss
                # pol_in = torch.cat([z_self, h_prev, lr_concat], dim=-1)  # (T,B,*)
                # pol_logits = self.self_policy_dec(pol_in)                # (T,B,2)

                # if self_evt.any():
                #     pol_loss = F.cross_entropy(pol_logits[self_evt], a_t[self_evt])
                # else:
                #     pol_loss = torch.zeros((), device=device)

                # self reward loss (only when reward is observed)
                sr_in = torch.cat([z_self, h_prev, lr_concat, chosen_bandits.float()], dim=-1)
                sr_logits = self.self_reward_dec(sr_in).squeeze(-1)  # (T,B)

                mask_self_r = self_evt & rwd_obs
                if mask_self_r.any():
                    self_rwd_loss = F.binary_cross_entropy_with_logits(sr_logits[mask_self_r], rwd01[mask_self_r])
                else:
                    self_rwd_loss = torch.zeros((), device=device)

                # -----------------------------
                # Decode: teacher behavior from z_obs
                # -----------------------------
                ta_in = torch.cat([z_obs, h_prev, lr_concat], dim=-1)
                teach_logits = self.teacher_action_dec(ta_in)  # (T,B,2)
                if teacher_evt.any():
                    teach_act_loss = F.cross_entropy(teach_logits[teacher_evt], a_t[teacher_evt])
                else:
                    teach_act_loss = torch.zeros((), device=device)

                # Optional teacher reward loss (only teacher events with observed reward; typically OL-o)
                tr_in = torch.cat([z_obs, h_prev, lr_concat, chosen_bandits.float()], dim=-1)
                tr_logits = self.teacher_reward_dec(tr_in).squeeze(-1)

                mask_teacher_r = teacher_evt & rwd_obs
                if mask_teacher_r.any():
                    teach_rwd_loss = F.binary_cross_entropy_with_logits(tr_logits[mask_teacher_r], rwd01[mask_teacher_r])
                else:
                    teach_rwd_loss = torch.zeros((), device=device)

                # -----------------------------
                # Total
                # -----------------------------
                beta_kl = 1.0
                # lambda_self_pol = 0.1
                lambda_teacher_act = 0.1
                lambda_teacher_rwd = 0.1

                bandit_total = (
                    self_rwd_loss
                    + lambda_teacher_act * teach_act_loss
                    + lambda_teacher_rwd * teach_rwd_loss
                    + beta_kl * kl_loss
                )

            bandit_total.backward()
            bandit_loss_sum += float(bandit_total.item())
            bandit_slices += 1

            # -----------------------------
            # 4) MOTOR loss (flat)
            # -----------------------------
            mini_batch_size = 16384
            total_steps = len(xy_pos_buf)
            for start in range(0, total_steps, mini_batch_size):
                end = min(start + mini_batch_size, total_steps)
                xy_slice = xy_pos[start:end]
                goal_slice = goalvec[start:end]
                chosen_slice = chosen_bandits_motor[start:end]

                mu, _log_std = self.motor_fwd(chosen_slice, xy_slice, goal_slice)

                dist = goal_slice.norm(dim=-1, keepdim=True) + 1e-6
                g_hat = goal_slice / dist
                speed = (dist / math.sqrt(8.0)).clamp(0.0, 1.0)
                target = (g_hat * speed).clamp(-0.999, 0.999)
                u_target = atanh(target)

                motor_loss = F.mse_loss(mu, u_target)
                motor_loss.backward()

                motor_loss_sum += float(motor_loss.item())
                motor_slices += 1

            torch.nn.utils.clip_grad_norm_(list(self.bandit_parameters()), 1.0)
            torch.nn.utils.clip_grad_norm_(list(self.motor_parameters()), 1.0)
            optim_bandit.step()
            optim_motor.step()

        bandit_loss = bandit_loss_sum / max(1, bandit_slices)
        motor_loss = motor_loss_sum / max(1, motor_slices)
        return bandit_loss, motor_loss