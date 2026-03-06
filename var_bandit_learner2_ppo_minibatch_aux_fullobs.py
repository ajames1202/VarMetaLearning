import math
from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """Small CNN for processing visual inputs (expects NCHW).

    The original version hard-coded the flatten size (32*14*14), which only
    works for one specific input resolution. This version uses
    AdaptiveAvgPool2d so it works for any HxW.
    """

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
    """Numerically-safe inverse tanh."""
    return 0.5 * (torch.log1p(x + 1e-6) - torch.log1p(-x + 1e-6))


def log_prob_tanh_normal(
    action: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Log-prob under a tanh-squashed Normal.

    Kept for backward compatibility with your original file.
    """
    std = torch.exp(log_std)
    # undo tanh
    pre = torch.atanh(action.clamp(-1 + eps, 1 - eps))
    normal = torch.distributions.Normal(mean, std)
    logp = normal.log_prob(pre).sum(dim=-1)
    # change-of-variables correction
    corr = torch.log(1 - torch.tanh(pre).pow(2) + eps).sum(dim=-1)
    return logp - corr


def report_param_set(params, label: str) -> None:
    """Utility: print how many params received non-zero grads."""
    params = list(params)
    touched = 0
    total = 0
    for p in params:
        total += 1
        if p.grad is not None and float(p.grad.abs().sum()) != 0.0:
            touched += 1
    print(f"{label}: touched {touched}/{total} params")


def debug_left_right_feats(left_feats: torch.Tensor, right_feats: torch.Tensor, tag: str = "") -> None:
    """Quick sanity check for encoder collapse vs random pairing baseline."""
    with torch.no_grad():
        lf = left_feats.reshape(-1, left_feats.size(-1))
        rf = right_feats.reshape(-1, right_feats.size(-1))
        lf_n = F.normalize(lf, dim=-1)
        rf_n = F.normalize(rf, dim=-1)

        cos_pair = (lf_n * rf_n).sum(dim=-1)
        l2_pair = (lf - rf).norm(dim=-1)

        perm = torch.randperm(rf.size(0), device=rf.device)
        cos_rand = (lf_n * rf_n[perm]).sum(dim=-1)
        l2_rand = (lf - rf[perm]).norm(dim=-1)

        print(f"[{tag}] cosine paired   mean/std: {cos_pair.mean():.3f} / {cos_pair.std():.3f}")
        print(f"[{tag}] cosine random   mean/std: {cos_rand.mean():.3f} / {cos_rand.std():.3f}")
        print(f"[{tag}] L2 paired       mean/std: {l2_pair.mean():.3f} / {l2_pair.std():.3f}")
        print(f"[{tag}] L2 random       mean/std: {l2_rand.mean():.3f} / {l2_rand.std():.3f}")


def lr_repulsion_loss(left_feats: torch.Tensor, right_feats: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    """Penalize left/right encodings being too similar (cosine similarity > margin)."""
    lf = F.normalize(left_feats.reshape(-1, left_feats.size(-1)), dim=-1)
    rf = F.normalize(right_feats.reshape(-1, right_feats.size(-1)), dim=-1)
    sim = (lf * rf).sum(dim=-1)
    return F.relu(sim - margin).mean()


class BanditLearner(nn.Module):
    """Bandit learner with
    - vision encoder
    - causal attention memory
    - reward baseline + residual corrections
    - motor head

    This file is a *fixed* version of the original:
    - removes torchvision dependency (torchvision import can fail in many envs)
    - fixes broken shapes (q_base/q_delta/gates)
    - removes undefined variables / duplicated dead code
    - makes attention BOS-safe when all keys are masked
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

        # Causal cross-attention over past trials.
        self.attn = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, dropout=0.0, batch_first=False)
        self.q_in = nn.Linear(feature_dim, rnn_hidden_size, bias=False)
        self.attn_ln = nn.LayerNorm(rnn_hidden_size)

        # BOS token prevents NaNs when key_padding_mask masks all keys for some batch elements.
        self.bos_token = nn.Parameter(torch.zeros(1, 1, rnn_hidden_size))

        # Event/value embeddings
        self.actor = nn.Embedding(2, rnn_hidden_size)   # 0=self, 1=teacher
        self.action = nn.Embedding(2, rnn_hidden_size)  # 0=left, 1=right
        self.rwd_in = nn.Embedding(3, rnn_hidden_size)  # 0,1,2(no_feedback)

        # Reward heads: return 1 logit per arm.
        self.q_base = nn.Sequential(
            nn.Linear(rnn_hidden_size + feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.q_delta_ao = nn.Sequential(
            nn.Linear(rnn_hidden_size + feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.q_delta_ol = nn.Sequential(
            nn.Linear(rnn_hidden_size + feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # Gates for residual strength.
        self.gate_ao = nn.Sequential(
            nn.Linear(4 * rnn_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.gate_ol = nn.Sequential(
            nn.Linear(4 * rnn_hidden_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # Motor policy
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
            self.q_in,
            self.attn_ln,
            self.actor,
            self.action,
            self.rwd_in,
            self.q_base,
            self.q_delta_ao,
            self.q_delta_ol,
            self.gate_ao,
            self.gate_ol,
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

    @staticmethod
    def _head(head: nn.Module, ctx: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """head([ctx, feats]) -> (T,B,1)"""
        return head(torch.cat([ctx, feats], dim=-1))

    def _safe_causal_attn(
        self,
        query: torch.Tensor,  # (L,B,H)
        key: torch.Tensor,    # (L,B,H)
        value: torch.Tensor,  # (L,B,H)
        key_padding_mask: torch.Tensor,  # (B,L) True=ignore
    ) -> torch.Tensor:
        """Causal attention with a BOS token so we never softmax over all-masked keys."""

        L, B, H = query.shape
        dev = query.device
        dtype = query.dtype
        

        bos_k = self.bos_token.expand(1, B, H)
        bos_v = torch.zeros((1, B, H), device=dev, dtype=dtype)

        key2 = torch.cat([bos_k, key], dim=0)   # (L+1,B,H)
        val2 = torch.cat([bos_v, value], dim=0)

        Lq, B, H = query.shape
        key_len = key.shape[0]          # past tokens
        src_len = key_len + 1           # + BOS


        bos_kpm = torch.zeros((B, 1), device=dev, dtype=torch.bool)
        kpm2 = torch.cat([bos_kpm, key_padding_mask], dim=1)  # (B,L+1)

        # Query i may attend to BOS and keys up to i (history).
        # With BOS prepended, disallow keys >= i+2.
        # attn_mask = torch.triu(torch.ones((L, L + 1), device=dev, dtype=torch.bool), diagonal=2)
        offset = max(0, key_len - Lq)
        attn_mask = torch.triu(
            torch.ones((Lq, src_len), device=dev, dtype=torch.bool),
            diagonal=offset + 2
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

    def _build_ctx(self, q_all: torch.Tensor, ctx_q: torch.Tensor) -> torch.Tensor:
        """Assemble full-length ctx with t=0 handled."""
        T, B, H = q_all.shape
        out = q_all.new_zeros(T, B, H)
        out[0] = self.attn_ln(q_all[0])
        if T > 1:
            out[1:] = ctx_q
        return out

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
        bandit_obs,
        chosen_bandits_buf,
        bandit_rewards_buf,
        meta_ep_start_buf,   # unused (kept for signature compat)
        actor_buf,
        trial_cond_buf,
        teacher_correct_buf,
        device,
        bandit_epochs: int = 4,
        max_grad_norm: float = 1.0,
        motor_epochs: int = 1,
        motor_minibatch_size: int = 16384,
        gate_coef: float = 0.1,
        contrast_coef: float = 0.05,
    ):
        """BCE update with probability-space AO/OL belief mixing and a separate motor update."""
        # ---------- motor data ----------
        xy_pos = torch.as_tensor(np.stack(xy_pos_buf), device=device, dtype=torch.float32)
        goalvec = torch.as_tensor(np.asarray(goal_vec_buf, np.float32), device=device)
        chosen_bandits_motor = torch.as_tensor(np.stack(chosen_bandits_motor_buf), device=device, dtype=torch.float32)

        # ---------- bandit data ----------
        left_obs = torch.stack([torch.as_tensor(b["left"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)
        right_obs = torch.stack([torch.as_tensor(b["right"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)

        chosen_bandits = torch.stack(chosen_bandits_buf, dim=0).to(device)     # (B,T,2)
        rewards_bandits = torch.stack(bandit_rewards_buf, dim=0).to(device)    # (B,T) 0/1/2 (2=no_feedback)
        actor = torch.stack(actor_buf, dim=0).to(device)                       # (B,T) 0=self 1=teacher
        teacher_correct = torch.stack(teacher_correct_buf, dim=0).to(device)   # (B,T) in {-1,0,1}
        trial_cond = np.stack(trial_cond_buf, axis=0)                          # (B,T) strings

        B, T, C, H, W = left_obs.shape

        # time-major
        chosen_bandits = chosen_bandits.permute(1, 0, 2)   # (T,B,2)
        rewards_bandits = rewards_bandits.permute(1, 0)    # (T,B)
        actor = actor.permute(1, 0).long().clamp(0, 1)     # (T,B)
        teacher_correct = teacher_correct.permute(1, 0).float()  # (T,B)
        trial_cond = trial_cond.transpose(1, 0)            # (T,B)

        actions = chosen_bandits.argmax(dim=-1)            # (T,B)
        rwd_idx = rewards_bandits.long().clamp(0, 2)       # (T,B)
        rwd_obs = (rwd_idx != 2)
        rwd01 = rwd_idx.float().clamp(0.0, 1.0)

        mask_IL  = torch.as_tensor(trial_cond == "IL",   device=device)
        mask_AOo = torch.as_tensor(trial_cond == "AO-o", device=device)
        mask_AOs = torch.as_tensor(trial_cond == "AO-s", device=device)
        mask_OLo = torch.as_tensor(trial_cond == "OL-o", device=device)
        mask_OLs = torch.as_tensor(trial_cond == "OL-s", device=device)
        mask_self_choice = (mask_IL | mask_AOs | mask_OLs)

        train_mask = mask_self_choice & rwd_obs
        if not train_mask.any():
            return 0.0, 0.0

        def forward_bandit():
            left_flat = left_obs.reshape(B * T, C, H, W)
            right_flat = right_obs.reshape(B * T, C, H, W)
            left_feats = self.encode(left_flat).view(B, T, -1).permute(1, 0, 2)
            right_feats = self.encode(right_flat).view(B, T, -1).permute(1, 0, 2)

            aL = chosen_bandits[..., 0:1]
            aR = chosen_bandits[..., 1:2]
            chosen_feat = aL * left_feats + aR * right_feats

            a_t = torch.argmax(chosen_bandits, dim=-1)
            x_val = self.action(a_t) + self.actor(actor) + self.rwd_in(rwd_idx)

            q_left_all = self.q_in(left_feats)
            q_right_all = self.q_in(right_feats)
            k_all = self.q_in(chosen_feat)

            zeros_logits = torch.zeros((T, B, 2), device=device, dtype=left_feats.dtype)
            zeros_gate = torch.zeros((T, B, 1), device=device, dtype=left_feats.dtype)
            zeros_scalar = torch.zeros((T, B), device=device, dtype=left_feats.dtype)
            if T < 2:
                eps = 1e-5
                p_self = torch.sigmoid(zeros_logits)
                return {
                    "q_self_logits": zeros_logits,
                    "q_teacher_ao": zeros_logits,
                    "q_teacher_ol": zeros_logits,
                    "g_ao": zeros_gate,
                    "g_ol": zeros_gate,
                    "ao_before": zeros_scalar,
                    "ol_before": zeros_scalar,
                    "q_mix_ao": torch.log(p_self.clamp(eps, 1.0 - eps) / (1.0 - p_self.clamp(eps, 1.0 - eps))),
                    "q_mix_ol": torch.log(p_self.clamp(eps, 1.0 - eps) / (1.0 - p_self.clamp(eps, 1.0 - eps))),
                    "left_feats": left_feats,
                    "right_feats": right_feats,
                }

            q_left_q = q_left_all[1:]
            q_right_q = q_right_all[1:]
            k_hist = k_all[:-1]
            v_hist = x_val[:-1]

            def cond_ctx(q_q, q_all, hist_keep_mask):
                kpm = ~(hist_keep_mask[:-1].T)
                ctx_q = self._safe_causal_attn(q_q, k_hist, v_hist, kpm)
                ctx_q = self.attn_ln(ctx_q + q_q)
                return self._build_ctx(q_all, ctx_q)

            ctx_left_il    = cond_ctx(q_left_q,  q_left_all,  mask_IL)
            ctx_right_il   = cond_ctx(q_right_q, q_right_all, mask_IL)
            ctx_left_ao_s  = cond_ctx(q_left_q,  q_left_all,  mask_AOs)
            ctx_right_ao_s = cond_ctx(q_right_q, q_right_all, mask_AOs)
            ctx_left_ol_s  = cond_ctx(q_left_q,  q_left_all,  mask_OLs)
            ctx_right_ol_s = cond_ctx(q_right_q, q_right_all, mask_OLs)
            ctx_left_ao_o  = cond_ctx(q_left_q,  q_left_all,  mask_AOo)
            ctx_right_ao_o = cond_ctx(q_right_q, q_right_all, mask_AOo)
            ctx_left_ol_o  = cond_ctx(q_left_q,  q_left_all,  mask_OLo)
            ctx_right_ol_o = cond_ctx(q_right_q, q_right_all, mask_OLo)

            base_l = (
                self._head(self.q_base, ctx_left_il, left_feats) * mask_IL.unsqueeze(-1).float()
                + self._head(self.q_base, ctx_left_ao_s, left_feats) * mask_AOs.unsqueeze(-1).float()
                + self._head(self.q_base, ctx_left_ol_s, left_feats) * mask_OLs.unsqueeze(-1).float()
            )
            base_r = (
                self._head(self.q_base, ctx_right_il, right_feats) * mask_IL.unsqueeze(-1).float()
                + self._head(self.q_base, ctx_right_ao_s, right_feats) * mask_AOs.unsqueeze(-1).float()
                + self._head(self.q_base, ctx_right_ol_s, right_feats) * mask_OLs.unsqueeze(-1).float()
            )
            q_self_logits = torch.cat([base_l, base_r], dim=-1)

            q_teacher_ao = torch.cat(
                [self._head(self.q_delta_ao, ctx_left_ao_o, left_feats),
                 self._head(self.q_delta_ao, ctx_right_ao_o, right_feats)], dim=-1
            )
            q_teacher_ol = torch.cat(
                [self._head(self.q_delta_ol, ctx_left_ol_o, left_feats),
                 self._head(self.q_delta_ol, ctx_right_ol_o, right_feats)], dim=-1
            )

            g_ao = torch.sigmoid(
                self.gate_ao(torch.cat([ctx_left_ao_s, ctx_right_ao_s, ctx_left_ao_o, ctx_right_ao_o], dim=-1))
            ).clamp(0.0, 1.0)
            g_ol = torch.sigmoid(
                self.gate_ol(torch.cat([ctx_left_ol_s, ctx_right_ol_s, ctx_left_ol_o, ctx_right_ol_o], dim=-1))
            ).clamp(0.0, 1.0)

            ao_cum = mask_AOo.float().cumsum(dim=0)
            ol_cum = mask_OLo.float().cumsum(dim=0)
            ao_before = torch.zeros_like(ao_cum)
            ol_before = torch.zeros_like(ol_cum)
            ao_before[1:] = ao_cum[:-1]
            ol_before[1:] = ol_cum[:-1]

            eps = 1e-5
            p_self = torch.sigmoid(q_self_logits.detach())
            p_teacher_ao = torch.sigmoid(q_teacher_ao)
            p_teacher_ol = torch.sigmoid(q_teacher_ol)
            p_mix_ao = ((1.0 - g_ao) * p_self + g_ao * p_teacher_ao).clamp(eps, 1.0 - eps)
            p_mix_ol = ((1.0 - g_ol) * p_self + g_ol * p_teacher_ol).clamp(eps, 1.0 - eps)
            q_mix_ao = torch.log(p_mix_ao / (1.0 - p_mix_ao))
            q_mix_ol = torch.log(p_mix_ol / (1.0 - p_mix_ol))

            return {
                "q_self_logits": q_self_logits,
                "q_teacher_ao": q_teacher_ao,
                "q_teacher_ol": q_teacher_ol,
                "g_ao": g_ao,
                "g_ol": g_ol,
                "ao_before": ao_before,
                "ol_before": ol_before,
                "q_mix_ao": q_mix_ao,
                "q_mix_ol": q_mix_ol,
                "left_feats": left_feats,
                "right_feats": right_feats,
            }

        bandit_loss_sum = 0.0
        bandit_steps = 0

        for _ in range(int(bandit_epochs)):
            optim_bandit.zero_grad(set_to_none=True)
            out = forward_bandit()

            q_self_logits = out["q_self_logits"]
            g_ao = out["g_ao"]
            g_ol = out["g_ol"]
            ao_before = out["ao_before"]
            ol_before = out["ol_before"]
            q_mix_ao = out["q_mix_ao"]
            q_mix_ol = out["q_mix_ol"]
            left_feats = out["left_feats"]
            right_feats = out["right_feats"]

            q_self_chosen = q_self_logits.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            base_loss = F.binary_cross_entropy_with_logits(q_self_chosen[train_mask], rwd01[train_mask])

            use_ao_mix = mask_AOs & (ao_before > 0) & rwd_obs
            use_ol_mix = mask_OLs & (ol_before > 0) & rwd_obs
            q_mix_ao_chosen = q_mix_ao.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            q_mix_ol_chosen = q_mix_ol.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            mix_losses = []
            if use_ao_mix.any():
                mix_losses.append(F.binary_cross_entropy_with_logits(q_mix_ao_chosen[use_ao_mix], rwd01[use_ao_mix]))
            if use_ol_mix.any():
                mix_losses.append(F.binary_cross_entropy_with_logits(q_mix_ol_chosen[use_ol_mix], rwd01[use_ol_mix]))
            mix_loss = torch.stack(mix_losses).mean() if mix_losses else torch.zeros((), device=device)

            tc = teacher_correct.clamp(0.0, 1.0)
            ao_sum = (tc * mask_AOo.float()).cumsum(dim=0)
            ao_cnt = mask_AOo.float().cumsum(dim=0)
            ol_sum = (tc * mask_OLo.float()).cumsum(dim=0)
            ol_cnt = mask_OLo.float().cumsum(dim=0)

            ao_rel = ao_sum / (ao_cnt + 1e-6)
            ol_rel = ol_sum / (ol_cnt + 1e-6)
            ao_rel_prev = torch.zeros_like(ao_rel)
            ol_rel_prev = torch.zeros_like(ol_rel)
            ao_rel_prev[1:] = ao_rel[:-1]
            ol_rel_prev[1:] = ol_rel[:-1]

            mask_gate_ao = mask_AOs & (ao_before > 0)
            mask_gate_ol = mask_OLs & (ol_before > 0)
            gate_losses = []
            if mask_gate_ao.any():
                gate_losses.append(F.mse_loss(g_ao.squeeze(-1)[mask_gate_ao], ao_rel_prev[mask_gate_ao]))
            if mask_gate_ol.any():
                gate_losses.append(F.mse_loss(g_ol.squeeze(-1)[mask_gate_ol], ol_rel_prev[mask_gate_ol]))
            gate_loss = torch.stack(gate_losses).mean() if gate_losses else torch.zeros((), device=device)

            q_bce = base_loss + mix_loss
            contrast_loss = lr_repulsion_loss(left_feats, right_feats, margin=0.2)
            bandit_total = q_bce + gate_coef * gate_loss + contrast_coef * contrast_loss

            bandit_total.backward()
            torch.nn.utils.clip_grad_norm_(list(self.bandit_parameters()), max_grad_norm)
            optim_bandit.step()

            bandit_loss_sum += float(q_bce.item())
            bandit_steps += 1

        bandit_loss = bandit_loss_sum / max(1, bandit_steps)

        # ---------- motor update (separate) ----------
        motor_loss_sum = 0.0
        motor_mb = 0
        total_steps = int(xy_pos.shape[0])

        for _ in range(int(motor_epochs)):
            perm = torch.randperm(total_steps, device=device)
            for start in range(0, total_steps, int(motor_minibatch_size)):
                idx = perm[start:start + int(motor_minibatch_size)]
                optim_motor.zero_grad(set_to_none=True)

                mu, _ = self.motor_fwd(chosen_bandits_motor[idx], xy_pos[idx], goalvec[idx])
                dist2 = goalvec[idx].norm(dim=-1, keepdim=True) + 1e-6
                g_hat = goalvec[idx] / dist2
                speed = (dist2 / math.sqrt(8.0)).clamp(0.0, 1.0)
                target = (g_hat * speed).clamp(-0.999, 0.999)
                u_target = atanh(target)

                motor_loss = F.mse_loss(mu, u_target)
                motor_loss.backward()

                torch.nn.utils.clip_grad_norm_(list(self.motor_parameters()), max_grad_norm)
                optim_motor.step()

                motor_loss_sum += float(motor_loss.item())
                motor_mb += 1

        motor_loss_out = motor_loss_sum / max(1, motor_mb)
        return bandit_loss, motor_loss_out
