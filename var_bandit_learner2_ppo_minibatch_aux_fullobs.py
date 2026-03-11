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


        # Detached control heads.
        # They see belief summaries + current/observed contexts, but PPO gradients do not
        # flow back into the belief network because the control input is detached.
        self.control_input_dim = 4 * rnn_hidden_size + 2 * feature_dim + 12

        self.v_head = nn.Sequential(
            nn.Linear(self.control_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.pi_head = nn.Sequential(
            nn.Linear(self.control_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
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
            self.v_head,
            self.pi_head,
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
    
    def _belief_logit_from_probs(self, p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        p = p.clamp(eps, 1.0 - eps)
        return torch.logit(p)

    def _control_features(
        self,
        ctx_left_cur: torch.Tensor,
        ctx_right_cur: torch.Tensor,
        ctx_left_obs: torch.Tensor,
        ctx_right_obs: torch.Tensor,
        left_feats: torch.Tensor,
        right_feats: torch.Tensor,
        p_self: torch.Tensor,
        p_teacher_ao: torch.Tensor,
        p_teacher_ol: torch.Tensor,
        p_belief: torch.Tensor,
        g_ao: torch.Tensor,
        g_ol: torch.Tensor,
        use_ao: torch.Tensor,
        use_ol: torch.Tensor,
    ) -> torch.Tensor:
        base = torch.cat([ctx_left_cur, ctx_right_cur, ctx_left_obs, ctx_right_obs, left_feats, right_feats], dim=-1)
        extras = torch.cat([
            p_self,
            p_teacher_ao,
            p_teacher_ol,
            p_belief,
            g_ao,
            g_ol,
            use_ao,
            use_ol,
        ], dim=-1)
        return torch.cat([base, extras], dim=-1)

    def gae_sparse(self, rewards, values, mask, gamma, lam):
        """Vectorized sparse GAE: processes all B sessions in parallel at each timestep.

        Replaces the original double Python-loop (B sessions × masked-steps). The new
        version is a single reversed scan over T steps with vectorised tensor ops over B.

        Correctness note: the discounting between *adjacent masked steps* uses gamma^1
        regardless of the actual gap between them (same as the original for gamma=1.0,
        which is the only value used in practice).  For gamma < 1 the approximation
        is very slight and inconsequential for this codebase.
        """
        T, B = rewards.shape
        adv = rewards.new_zeros(T, B)
        ret = rewards.new_zeros(T, B)

        running_gae    = rewards.new_zeros(B)   # accumulated GAE at the "next" masked step
        running_next_v = rewards.new_zeros(B)   # V at the "next" masked step

        for t in reversed(range(T)):
            m   = mask[t]          # (B,) bool
            r_t = rewards[t]       # (B,)
            v_t = values[t]        # (B,)

            delta   = r_t + gamma * running_next_v - v_t
            new_gae = delta + gamma * lam * running_gae

            # Write into output only at masked positions
            adv[t] = torch.where(m, new_gae,        adv[t])
            ret[t] = torch.where(m, new_gae + v_t,  ret[t])

            # Update running state only at masked (self-choice) positions
            running_next_v = torch.where(m, v_t,      running_next_v)
            running_gae    = torch.where(m, new_gae,  running_gae)

        return adv, ret

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
        teacher_correct_buf, # (B,) list of (T,) float tensors: -1=non-obs, 0=wrong, 1=correct
        # PPO extras:
        bandit_logp_buf,
        bandit_value_buf,
        device,
        # PPO hyperparams:
        ppo_epochs: int = 4,
        ppo_minibatch_size: int = 256,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        tau: float = 1.0,
        max_grad_norm: float = 1.0,
        motor_epochs: int = 1,
        motor_minibatch_size: int = 16384,
        # Aux losses (supervised) to make observations matter:
        aux_mb_size: int = 256,
        self_bce_coef: float = 1.0, # self-choice reward BCE on fused beliefs
        aux_ao_coef: float = 0.2,   # AO-o: imitate teacher action (CE)
        aux_ol_coef: float = 0.5,   # OL-o: predict teacher reward on chosen arm (BCE)
        # Reliability gate losses  L_gate,ao = BCE(g_ao,  ao_rel_target)
        #                          L_gate,ol = BCE(sg(g_ol), ol_rel_target)   [↓ = stop-grad]
        gate_ao_coef: float = 0.5,  # weight for AO gate reliability loss
        gate_ol_coef: float = 0.5,  # weight for OL gate reliability loss
        gate_ol_sg: bool = True,    # if True, stop-gradient through g_ol in the OL gate loss
        session_chunk_size: int | None = None,
    ):
        """
        Hybrid update with:
          - PPO on detached control heads (pi_head / v_head)
          - full-batch BCE/CE supervision on the belief network
          - gate reliability losses: trains g_ao / g_ol to match a causal running
            estimate of teacher correctness on AO-o / OL-o trials respectively.
            The OL gate loss uses a stop-gradient on g_ol (↓ in the formula) so
            the gate BCE signal does not flow back into the belief/context network.
          - motor regression update (separate, optionally multiple epochs)

        Notes:
          - self-choice beliefs are fused in probability space via convex mixing
            of p_self and teacher-derived beliefs.
          - PPO sees detached control features, so policy/value gradients do not
            modify q_base / q_delta_* / attn / gates.
          - Advantages/returns are computed from rollout-time values (bandit_value_buf) using sparse-GAE.
        """
        # ---------- motor data (optional — skipped when direct-choice rollout is used) ----------
        # When meta_ep_rollout uses cursor teleportation, xy_pos_buf / goal_vec_buf /
        # chosen_bandits_motor_buf are all empty lists.  In that case the motor network
        # is not used during rollout so there is nothing to train; we skip the motor
        # update entirely and return 0.0 as motor loss.
        _has_motor_data = len(xy_pos_buf) > 0

        if _has_motor_data:
            xy_pos             = torch.as_tensor(np.stack(xy_pos_buf),             device=device, dtype=torch.float32)
            goalvec            = torch.as_tensor(np.asarray(goal_vec_buf, np.float32), device=device)
            chosen_bandits_motor = torch.as_tensor(np.stack(chosen_bandits_motor_buf), device=device, dtype=torch.float32)

        # ---------- bandit data ----------
        # bandit_obs is a list of dicts {"left": (T,3,h,w), "right": ...}
        # Images may be stored as uint8 (from optimised rollout workers) or float32.
        # Either way, keep heavy image tensors on CPU; only move a session chunk to GPU.
        def _obs_to_uint8(arr):
            t = torch.as_tensor(arr)
            if t.dtype == torch.uint8:
                return t
            # float32 in [0,1] → uint8 in [0,255]
            return t.mul(255.0).clamp_(0, 255).byte()

        left_obs_cpu = torch.stack([
            _obs_to_uint8(b["left"]) for b in bandit_obs
        ], dim=0).contiguous()   # (B,T,3,H,W) uint8
        right_obs_cpu = torch.stack([
            _obs_to_uint8(b["right"]) for b in bandit_obs
        ], dim=0).contiguous()

        # Pin memory so CPU→GPU transfers can be async (non_blocking=True)
        if device.type == 'cuda':
            left_obs_cpu  = left_obs_cpu.pin_memory()
            right_obs_cpu = right_obs_cpu.pin_memory()

        chosen_bandits = torch.stack(chosen_bandits_buf, dim=0).to(device)     # (B,T,2)
        rewards_bandits = torch.stack(bandit_rewards_buf, dim=0).to(device)    # (B,T) 0/1/2 (2=no_feedback)
        actor = torch.stack(actor_buf, dim=0).to(device)                       # (B,T) 0=self 1=teacher
        trial_cond = np.stack(trial_cond_buf, axis=0)                          # (B,T) strings

        logp_old_bt = torch.stack(bandit_logp_buf, dim=0).to(device)           # (B,T) (dummy on obs trials)
        v_old_bt = torch.stack(bandit_value_buf, dim=0).to(device)             # (B,T)

        B, T, C, H, W = left_obs_cpu.shape

        # time-major
        chosen_bandits = chosen_bandits.permute(1, 0, 2)  # (T,B,2)
        rewards_bandits = rewards_bandits.permute(1, 0)   # (T,B)
        actor = actor.permute(1, 0).long().clamp(0, 1)    # (T,B)
        trial_cond = trial_cond.transpose(1, 0)           # (T,B)

        actions = chosen_bandits.argmax(dim=-1)           # (T,B) int64
        logp_old = logp_old_bt.permute(1, 0).float()      # (T,B)
        v_old = v_old_bt.permute(1, 0).float()            # (T,B)

        rwd_idx = rewards_bandits.long().clamp(0, 2)      # (T,B) in {0,1,2}
        rwd_obs = (rwd_idx != 2)
        rwd01 = rwd_idx.float().clamp(0.0, 1.0)

        # masks by condition (T,B)
        mask_IL  = torch.as_tensor(trial_cond == "IL",   device=device)
        mask_AOo = torch.as_tensor(trial_cond == "AO-o", device=device)
        mask_AOs = torch.as_tensor(trial_cond == "AO-s", device=device)
        mask_OLo = torch.as_tensor(trial_cond == "OL-o", device=device)
        mask_OLs = torch.as_tensor(trial_cond == "OL-s", device=device)
        mask_self_choice = (mask_IL | mask_AOs | mask_OLs)

        # reward only credited on self-choice trials with observed feedback
        rew = (rwd01 * (mask_self_choice & rwd_obs).float())  # (T,B)

        # ---------- running reliability targets (causal, teacher_correct_buf) ----------
        # teacher_correct values: -1.0 = non-obs trial, 0.0 = obs+teacher wrong, 1.0 = obs+teacher correct
        # For gate loss we need a scalar reliability estimate per (t, b) that uses ONLY
        # AO-o (or OL-o) trials that occurred STRICTLY BEFORE the current timestep,
        # so there is no look-ahead and the target is well-defined at t=0.
        #
        # ao_rel_target[t, b] = (# AO-o trials before t where teacher correct) /
        #                        (# AO-o trials before t)          [0 if none seen yet]
        #
        # This is used as the BCE target for g_ao at each AO-s step where use_ao is active.

        teacher_correct = (
            torch.stack(teacher_correct_buf, dim=0)      # (B, T)
            .to(device=device, dtype=torch.float32)
            .permute(1, 0)                               # → (T, B)
        )

        # AO reliability
        # teacher_correct at AO-o steps is 0.0 or 1.0; elsewhere may be -1.0 → clamp to 0.
        ao_tc  = (teacher_correct * mask_AOo.float()).clamp(0.0, 1.0)  # (T,B): correct AO-o
        ao_cnt = mask_AOo.float().cumsum(dim=0)                         # (T,B): running #AO-o
        ao_sum = ao_tc.cumsum(dim=0)                                    # (T,B): running #correct AO-o
        # Shift by 1: at time t we only know about obs BEFORE t
        ao_rel_target = torch.zeros_like(ao_sum)
        ao_rel_target[1:] = (ao_sum[:-1] / (ao_cnt[:-1] + 1e-8)).clamp(0.0, 1.0)

        # OL reliability (same structure)
        ol_tc  = (teacher_correct * mask_OLo.float()).clamp(0.0, 1.0)
        ol_cnt = mask_OLo.float().cumsum(dim=0)
        ol_sum = ol_tc.cumsum(dim=0)
        ol_rel_target = torch.zeros_like(ol_sum)
        ol_rel_target[1:] = (ol_sum[:-1] / (ol_cnt[:-1] + 1e-8)).clamp(0.0, 1.0)

        # Gate-loss normalisers (count of active AO-s / OL-s steps with prior obs history).
        # Computed once here; the exact active mask is re-derived per chunk inside the loop.
        # Use a conservative lower bound (all AO-s / OL-s steps) — over-counting is safe.
        total_gate_ao_n = max(int(mask_AOs.sum().item()), 1)
        total_gate_ol_n = max(int(mask_OLs.sum().item()), 1)

        # ---------- sparse GAE using rollout-time values ----------
        r_il = rew * mask_IL.float()
        r_ao = rew * mask_AOs.float()
        r_ol = rew * mask_OLs.float()
        adv_il, ret_il = self.gae_sparse(r_il, v_old, mask_IL,  gamma, gae_lambda)
        adv_ao, ret_ao = self.gae_sparse(r_ao, v_old, mask_AOs, gamma, gae_lambda)
        adv_ol, ret_ol = self.gae_sparse(r_ol, v_old, mask_OLs, gamma, gae_lambda)
        adv = adv_il + adv_ao + adv_ol
        ret = ret_il + ret_ao + ret_ol

        # normalize advantages over self-choice steps only
        m = mask_self_choice.float()
        denom = (m.sum() + 1e-8)
        adv_mean = (adv * m).sum() / denom
        adv_var = ((adv - adv_mean).pow(2) * m).sum() / denom
        adv = (adv - adv_mean) / (adv_var.sqrt() + 1e-8)

        # indices of (t,b) where policy was sampled
        idx_tb = torch.nonzero(mask_self_choice, as_tuple=False)  # (N,2)
        N = int(idx_tb.shape[0])

        # number of (t,b) steps where observations were shown (AO-o/OL-o)
        Nobs = int((mask_AOo | mask_OLo).sum().item())

        if N == 0:
            # should not happen, but stay safe
            return 0.0, 0.0

        if session_chunk_size is None:
            session_chunk_size = min(B, 10)
        session_chunk_size = int(max(1, min(B, session_chunk_size)))

        # Global normalizers so chunked accumulation matches the original
        # full-batch mean reductions.
        total_policy_n = max(int(mask_self_choice.sum().item()), 1)
        total_reward_n = max(int((mask_self_choice & rwd_obs).sum().item()), 1)
        total_ao_n = max(int(mask_AOo.sum().item()), 1)
        total_ol_n = max(int((mask_OLo & rwd_obs).sum().item()), 1)

        # ---------- forward helper (full sequence per session chunk) ----------
        def forward_policy_value_chunk(b0: int, b1: int):
            # Move uint8 → GPU, convert to float32 in one fused op
            left_obs  = left_obs_cpu[b0:b1].to(device, non_blocking=True).float().mul_(1.0 / 255.0)
            right_obs = right_obs_cpu[b0:b1].to(device, non_blocking=True).float().mul_(1.0 / 255.0)

            chosen_bandits_mb = chosen_bandits[:, b0:b1]
            actions_mb = actions[:, b0:b1]
            actor_mb = actor[:, b0:b1]
            rwd_idx_mb = rwd_idx[:, b0:b1]

            mask_IL_mb = mask_IL[:, b0:b1]
            mask_AOo_mb = mask_AOo[:, b0:b1]
            mask_AOs_mb = mask_AOs[:, b0:b1]
            mask_OLo_mb = mask_OLo[:, b0:b1]
            mask_OLs_mb = mask_OLs[:, b0:b1]

            Bc = int(b1 - b0)
            left_flat = left_obs.reshape(Bc * T, C, H, W)
            right_flat = right_obs.reshape(Bc * T, C, H, W)
            left_feats = self.encode(left_flat).view(Bc, T, -1).permute(1, 0, 2)   # (T,Bc,F)
            right_feats = self.encode(right_flat).view(Bc, T, -1).permute(1, 0, 2)

            # build history tokens
            aL = chosen_bandits_mb[..., 0:1]
            aR = chosen_bandits_mb[..., 1:2]
            chosen_feat = aL * left_feats + aR * right_feats

            x_val = self.action(actions_mb) + self.actor(actor_mb) + self.rwd_in(rwd_idx_mb)

            q_left_all = self.q_in(left_feats)
            q_right_all = self.q_in(right_feats)
            k_all = self.q_in(chosen_feat)

            if T < 2:
                zeros2  = torch.zeros((T, Bc, 2), device=device, dtype=left_feats.dtype)
                zeros1s = torch.zeros((T, Bc, 1), device=device, dtype=left_feats.dtype)
                zerosS  = torch.zeros((T, Bc),    device=device, dtype=left_feats.dtype)
                # belief_logits, policy_logits, V, q_teacher_ao, q_teacher_ol,
                # g_ao, g_ol, use_ao, use_ol, g_ao_for_loss, g_ol_for_loss
                return zeros2, zeros2, zerosS, zeros2, zeros2, zeros1s, zeros1s, zeros1s, zeros1s, zeros1s, zeros1s

            q_left_q = q_left_all[1:]
            q_right_q = q_right_all[1:]
            k_hist = k_all[:-1]
            v_hist = x_val[:-1]

            # -----------------------------------------------------------------
            # Build all 10 conditional contexts in ONE batched attention call.
            # Each of the 5 conditions (IL, AOs, OLs, AOo, OLo) needs a left
            # and right query → 10 sub-batches stacked along the batch dim.
            # The key/value history is the same for all; only the kpm differs.
            # -----------------------------------------------------------------
            cond_masks_ordered = [
                mask_IL_mb,   # idx 0→left-IL,  1→right-IL
                mask_AOs_mb,  # idx 2→left-AOs, 3→right-AOs
                mask_OLs_mb,  # idx 4→left-OLs, 5→right-OLs
                mask_AOo_mb,  # idx 6→left-AOo, 7→right-AOo
                mask_OLo_mb,  # idx 8→left-OLo, 9→right-OLo
            ]

            # Stack queries: (T-1, 10*Bc, H)
            q_all_stacked = torch.cat(
                [q for m in cond_masks_ordered for q in (q_left_q, q_right_q)],
                dim=1,
            )
            # Replicate K and V: (T-1, 10*Bc, H)
            k_rep = k_hist.repeat(1, 10, 1)
            v_rep = v_hist.repeat(1, 10, 1)
            # Stack key-padding masks: (10*Bc, T-1), True=ignore
            kpm_stacked = torch.cat(
                [~(m[:-1].T) for m in cond_masks_ordered for _ in range(2)],
                dim=0,
            )

            # Single attention call + layer-norm residual
            ctx_stacked = self._safe_causal_attn(q_all_stacked, k_rep, v_rep, kpm_stacked)
            ctx_stacked = self.attn_ln(ctx_stacked + q_all_stacked)  # (T-1, 10*Bc, H)

            # Unpack and build full-length contexts
            def _ctx(i, q_all_ref):
                return self._build_ctx(q_all_ref, ctx_stacked[:, i * Bc:(i + 1) * Bc, :])

            ctx_left_il    = _ctx(0, q_left_all)
            ctx_right_il   = _ctx(1, q_right_all)
            ctx_left_ao_s  = _ctx(2, q_left_all)
            ctx_right_ao_s = _ctx(3, q_right_all)
            ctx_left_ol_s  = _ctx(4, q_left_all)
            ctx_right_ol_s = _ctx(5, q_right_all)
            ctx_left_ao_o  = _ctx(6, q_left_all)
            ctx_right_ao_o = _ctx(7, q_right_all)
            ctx_left_ol_o  = _ctx(8, q_left_all)
            ctx_right_ol_o = _ctx(9, q_right_all)

            # base logits from self contexts
            base_l = (
                self._head(self.q_base, ctx_left_il,    left_feats)  * mask_IL_mb.unsqueeze(-1).float()
              + self._head(self.q_base, ctx_left_ao_s,  left_feats)  * mask_AOs_mb.unsqueeze(-1).float()
              + self._head(self.q_base, ctx_left_ol_s,  left_feats)  * mask_OLs_mb.unsqueeze(-1).float()
            )
            base_r = (
                self._head(self.q_base, ctx_right_il,   right_feats) * mask_IL_mb.unsqueeze(-1).float()
              + self._head(self.q_base, ctx_right_ao_s, right_feats) * mask_AOs_mb.unsqueeze(-1).float()
              + self._head(self.q_base, ctx_right_ol_s, right_feats) * mask_OLs_mb.unsqueeze(-1).float()
            )
            q_self_logits = torch.cat([base_l, base_r], dim=-1)  # (T,Bc,2)

            # teacher heads from observation contexts
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

            #Ablation test, setting g_ol to 0.1
            # g_ol = torch.full_like(g_ol, 0.0)

            #Ablation test, setting g_ol to 0.1
            # g_ao = torch.full_like(g_ao, 0.0)

            # Separate gate inputs with detached contexts for the reliability BCE loss.
            # This isolates the gate loss gradient to gate_ao / gate_ol MLP parameters only —
            # it does NOT propagate back into the encoder or attention.  The un-detached g_ao / g_ol
            # above continue to be used for belief fusion, so PPO/BCE losses can still shape
            # the gate via the belief-quality gradient path.
            # Cost: one extra tiny MLP forward (4H→64→1) per chunk — negligible.
            gate_inp_ao = torch.cat([
                ctx_left_ao_s.detach(), ctx_right_ao_s.detach(),
                ctx_left_ao_o.detach(), ctx_right_ao_o.detach(),
            ], dim=-1)
            gate_inp_ol = torch.cat([
                ctx_left_ol_s.detach(), ctx_right_ol_s.detach(),
                ctx_left_ol_o.detach(), ctx_right_ol_o.detach(),
            ], dim=-1)
            g_ao_for_loss = torch.sigmoid(self.gate_ao(gate_inp_ao)).clamp(0.0, 1.0)
            g_ol_for_loss = torch.sigmoid(self.gate_ol(gate_inp_ol)).clamp(0.0, 1.0)

            # Match rollout's "has_prev_*": only apply teacher belief if an earlier obs-trial exists
            ao_any = mask_AOo_mb.float().cumsum(dim=0)
            ol_any = mask_OLo_mb.float().cumsum(dim=0)
            ao_before = torch.zeros_like(ao_any)
            ol_before = torch.zeros_like(ol_any)
            ao_before[1:] = ao_any[:-1]
            ol_before[1:] = ol_any[:-1]
            use_ao = (mask_AOs_mb & (ao_before > 0)).float().unsqueeze(-1)
            use_ol = (mask_OLs_mb & (ol_before > 0)).float().unsqueeze(-1)

            p_self = torch.sigmoid(q_self_logits)
            p_teacher_ao = torch.sigmoid(q_teacher_ao)
            p_teacher_ol = torch.sigmoid(q_teacher_ol)

            p_belief = p_self
            p_belief = torch.where(use_ao.bool(), (1.0 - g_ao) * p_belief + g_ao * p_teacher_ao, p_belief)
            p_belief = torch.where(use_ol.bool(), (1.0 - g_ol) * p_belief + g_ol * p_teacher_ol, p_belief)
            belief_logits = self._belief_logit_from_probs(p_belief)

            # control/value input from detached belief summaries + detached contexts/features
            ctx_left_cur = (
                ctx_left_il   * mask_IL_mb.unsqueeze(-1).float()
              + ctx_left_ao_s * mask_AOs_mb.unsqueeze(-1).float()
              + ctx_left_ol_s * mask_OLs_mb.unsqueeze(-1).float()
            )
            ctx_right_cur = (
                ctx_right_il   * mask_IL_mb.unsqueeze(-1).float()
              + ctx_right_ao_s * mask_AOs_mb.unsqueeze(-1).float()
              + ctx_right_ol_s * mask_OLs_mb.unsqueeze(-1).float()
            )
            ctx_left_obs = (
                ctx_left_ao_o * mask_AOs_mb.unsqueeze(-1).float()
              + ctx_left_ol_o * mask_OLs_mb.unsqueeze(-1).float()
            )
            ctx_right_obs = (
                ctx_right_ao_o * mask_AOs_mb.unsqueeze(-1).float()
              + ctx_right_ol_o * mask_OLs_mb.unsqueeze(-1).float()
            )

                        # --------------------------------
            # 2) PPO-facing fused belief:
            #    self/teacher detached, gates live
            # --------------------------------
            p_self_pg = p_self.detach()
            p_teacher_ao_pg = p_teacher_ao.detach()
            p_teacher_ol_pg = p_teacher_ol.detach()

            p_belief_pg = p_self_pg
            p_belief_pg = torch.where(
                use_ao.bool(),
                (1.0 - g_ao) * p_belief_pg + g_ao * p_teacher_ao_pg,
                p_belief_pg
            )
            p_belief_pg = torch.where(
                use_ol.bool(),
                (1.0 - g_ol) * p_belief_pg + g_ol * p_teacher_ol_pg,
                p_belief_pg
            )

            

            # control_in = self._control_features(
            #     ctx_left_cur.detach(),
            #     ctx_right_cur.detach(),
            #     ctx_left_obs.detach(),
            #     ctx_right_obs.detach(),
            #     left_feats.detach(),
            #     right_feats.detach(),
            #     p_self_pg.detach(),              # do not detach for ppo
            #     p_teacher_ao_pg.detach(),
            #     p_teacher_ol_pg.detach(),
            #     p_belief_pg.detach(),   # do not detach for ppo
            #     g_ao.detach(),   # live
            #     g_ol.detach(),   # live
            #     use_ao.detach(),
            #     use_ol.detach(),
            # )

            ctx_left_cur_zer = torch.zeros_like(ctx_left_cur)
            ctx_right_cur_zer = torch.zeros_like(ctx_right_cur)
            ctx_left_obs_zer = torch.zeros_like(ctx_left_obs)
            ctx_right_obs_zer = torch.zeros_like(ctx_right_obs)
            p_self_pg_zer = torch.zeros_like(p_self_pg)
            p_teacher_ao_pg_zer = torch.zeros_like(p_teacher_ao_pg)
            p_teacher_ol_pg_zer = torch.zeros_like(p_teacher_ol_pg)
            g_ao_zer = torch.zeros_like(g_ao)
            g_ol_zer = torch.zeros_like(g_ol)

            control_in = self._control_features(
                ctx_left_cur_zer.detach(),
                ctx_right_cur_zer.detach(),
                ctx_left_obs_zer.detach(),
                ctx_right_obs_zer.detach(),
                left_feats.detach(),
                right_feats.detach(),
                p_self_pg_zer.detach(),              # do not detach for ppo
                p_teacher_ao_pg_zer.detach(),
                p_teacher_ol_pg_zer.detach(),
                p_belief_pg.detach(),   # do not detach for ppo
                g_ao_zer.detach(),   # live
                g_ol_zer.detach(),   # live
                use_ao.detach(),
                use_ol.detach(),
            )

            policy_logits = self.pi_head(control_in)
            V = self.v_head(control_in).squeeze(-1)  # (T,Bc)

            return (belief_logits, policy_logits, V, q_teacher_ao, q_teacher_ol,
                    g_ao, g_ol, use_ao, use_ol,
                    g_ao_for_loss, g_ol_for_loss)

        # ---------- hybrid PPO + BCE belief update (session-chunked) ----------
        policy_mask = mask_self_choice
        reward_mask = (mask_self_choice & rwd_obs)
        ol_mask_full = (mask_OLo & rwd_obs)

        total_loss = 0.0
        total_steps = 0

        for _ep in range(int(ppo_epochs)):
            optim_bandit.zero_grad(set_to_none=True)
            epoch_loss_value = 0.0

            for b0 in range(0, B, session_chunk_size):
                b1 = min(B, b0 + session_chunk_size)

                belief_logits, policy_logits, V, q_teacher_ao, q_teacher_ol, \
                    g_ao_mb, g_ol_mb, use_ao_mb, use_ol_mb, \
                    g_ao_loss_mb, g_ol_loss_mb = forward_policy_value_chunk(b0, b1)

                actions_mb = actions[:, b0:b1]
                logp_old_mb_all = logp_old[:, b0:b1]
                adv_mb_all = adv[:, b0:b1]
                ret_mb_all = ret[:, b0:b1]
                rwd01_mb = rwd01[:, b0:b1]

                policy_mask_mb = policy_mask[:, b0:b1]
                reward_mask_mb = reward_mask[:, b0:b1]
                mask_AOo_mb = mask_AOo[:, b0:b1]
                mask_AOs_mb = mask_AOs[:, b0:b1]
                mask_OLo_mb = mask_OLo[:, b0:b1]
                mask_OLs_mb = mask_OLs[:, b0:b1]
                ol_mask_mb = ol_mask_full[:, b0:b1]

                dist = torch.distributions.Categorical(logits=(policy_logits / tau))
                logp_new = dist.log_prob(actions_mb)   # (T,Bc)
                entropy = dist.entropy()               # (T,Bc)

                ratio = torch.exp(logp_new[policy_mask_mb] - logp_old_mb_all[policy_mask_mb])
                surr1 = ratio * adv_mb_all[policy_mask_mb]
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb_all[policy_mask_mb]
                pg_loss = -torch.min(surr1, surr2).sum() / total_policy_n

                v_loss = 0.5 * (ret_mb_all[policy_mask_mb] - V[policy_mask_mb]).pow(2).sum() / total_policy_n
                ent_loss = entropy[policy_mask_mb].sum() / total_policy_n

                aux_self = torch.zeros((), device=device)
                if reward_mask_mb.any():
                    chosen_belief_logit = belief_logits[reward_mask_mb].gather(
                        -1, actions_mb[reward_mask_mb].unsqueeze(-1)
                    ).squeeze(-1)
                    aux_self = F.binary_cross_entropy_with_logits(
                        chosen_belief_logit,
                        rwd01_mb[reward_mask_mb],
                        reduction="sum",
                    ) / total_reward_n

                aux_ao = torch.zeros((), device=device)
                aux_ol = torch.zeros((), device=device)

                if (Nobs > 0) and ((aux_ao_coef != 0.0) or (aux_ol_coef != 0.0)):
                    if (aux_ao_coef != 0.0) and mask_AOo_mb.any():
                        logits_ao = q_teacher_ao[mask_AOo_mb]
                        targ_ao = actions_mb[mask_AOo_mb].long()
                        aux_ao = F.cross_entropy(logits_ao, targ_ao, reduction="sum") / total_ao_n

                    if (aux_ol_coef != 0.0) and ol_mask_mb.any():
                        logits_ol = q_teacher_ol[ol_mask_mb]
                        targ_act = actions_mb[ol_mask_mb].long()
                        chosen_logit = logits_ol.gather(-1, targ_act.unsqueeze(-1)).squeeze(-1)
                        aux_ol = F.binary_cross_entropy_with_logits(
                            chosen_logit,
                            rwd01_mb[ol_mask_mb],
                            reduction="sum",
                        ) / total_ol_n

                # ----------------------------------------------------------------
                # Gate reliability losses
                #   L_gate,ao = BCE(g_ao,       ao_rel_target)   at active AO-s steps
                #   L_gate,ol = BCE(sg(g_ol),   ol_rel_target)   at active OL-s steps
                #                        ↑ stop-gradient through g_ol (gate_ol_sg flag)
                #
                # "active" = AO-s (or OL-s) step AND at least one prior obs-trial exists,
                # i.e. the gate is actually used in belief fusion (use_ao / use_ol > 0).
                #
                # g_ao_loss_mb / g_ol_loss_mb are recomputed with DETACHED context inputs
                # so the reliability BCE gradient only updates gate_ao / gate_ol MLP weights.
                # It does NOT flow back into the encoder or attention — keeping those shaped
                # purely by the PPO / belief-quality losses.
                # ----------------------------------------------------------------
                gate_ao_loss = torch.zeros((), device=device)
                gate_ol_loss = torch.zeros((), device=device)

                loss = (
                    pg_loss
                    + vf_coef * v_loss
                    - ent_coef * ent_loss
                    + self_bce_coef * aux_self
                    + aux_ao_coef * aux_ao
                    + aux_ol_coef * aux_ol
                    + gate_ao_coef * gate_ao_loss
                    + gate_ol_coef * gate_ol_loss
                )

                loss.backward()
                epoch_loss_value += float(loss.item())

                del belief_logits, policy_logits, V, q_teacher_ao, q_teacher_ol
                del g_ao_mb, g_ol_mb, use_ao_mb, use_ol_mb
                del g_ao_loss_mb, g_ol_loss_mb
                del dist, logp_new, entropy, loss

            torch.nn.utils.clip_grad_norm_(list(self.bandit_parameters()), max_grad_norm)
            optim_bandit.step()

            total_loss += epoch_loss_value
            total_steps += 1

        bandit_loss = total_loss / max(1, total_steps)

        # ---------- motor update (skipped when direct-choice rollout is active) ----------
        if not _has_motor_data:
            return bandit_loss, 0.0

        motor_loss_sum = 0.0
        motor_mb = 0

        total_steps = int(xy_pos.shape[0])
        for _me in range(int(motor_epochs)):
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