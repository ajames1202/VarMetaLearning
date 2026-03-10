# bandit_train_batch.py
# Train + evaluate IL/AO/OL bandit agent.
# Includes:
#   - early-stopped training loop
#   - fixed-prob eval + per-trial plot
#   - gap sweep eval: p_hi fixed, p_lo swept
#
# Notes:
#   - This file is designed to be backward-compatible with older var_bandit_learner2.py
#     (no teacher_correct_buf argument). If your learner supports teacher_correct_buf,
#     it will be passed; otherwise it will be ignored.

import argparse
import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import ray

import visual_bandit_env3 as vbe

import var_bandit_learner2_ppo_minibatch_aux_fullobs as bl

from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import time



# -----------------------------
# Utils
# -----------------------------

def extract_lr_views(obs_tensor: torch.Tensor, env, crop_size: int = 112, pad: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop left/right bandit patches from the full-frame observation and resize to (crop_size,crop_size)."""
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


def save_checkpoint(path: str, agent, optim_bandit, optim_motor, extra: Dict[str, Any]) -> None:
    ckpt = {
        "model_state": agent.state_dict(),
        "optim_bandit_state": optim_bandit.state_dict(),
        "optim_motor_state": optim_motor.state_dict(),
        "extra": extra,
        "torch": torch.__version__,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def sample_probs_this_session(session_K: int, p_hi: float, p_lo: float, rng: np.random.Generator) -> List[Tuple[float, float]]:
    """Randomize which side has higher prob per pair."""
    probs = []
    for _ in range(session_K):
        if rng.random() < 0.5:
            probs.append((p_hi, p_lo))
        else:
            probs.append((p_lo, p_hi))
    return probs



# -----------------------------
# Grid eval for early stopping
# -----------------------------

# Default training/eval grid (can edit here)
P_LO_GRID_DEFAULT = [0.2, 0.3, 0.4, 0.5, 0.6]

_DEFAULT_TEACHER_MODES = [
    ["expert", {"expert_teacher": True, "unrealiable_teacher": False, "eps": 0.10}],
]

_raw_teacher_modes = json.loads(
    os.environ.get("TEACHER_MODES_DEFAULT", json.dumps(_DEFAULT_TEACHER_MODES))
)

TEACHER_MODES_DEFAULT = [(name, cfg) for name, cfg in _raw_teacher_modes]


num_grid_cells = len(P_LO_GRID_DEFAULT) * len(TEACHER_MODES_DEFAULT)
# Keys used by teacher configs; used to reset env attrs each session to avoid cross-contamination.
TEACHER_CFG_KEYS = sorted({k for _, cfg in TEACHER_MODES_DEFAULT for k in cfg.keys()})


def _ray_get_in_batches(futures, batch_size: int):
    """Fetch Ray results in chunks to avoid large temporary result piles on the driver."""
    out = []
    batch_size = max(1, int(batch_size))
    for i in range(0, len(futures), batch_size):
        out.extend(ray.get(futures[i:i + batch_size]))
    return out


def eval_grid_score(
    workers,
    agent_state_cpu,
    session_K: int,
    p_hi: float,
    p_lo_grid,
    teacher_modes,
    n_sessions_per_cell: int,
    seed: int,
    alpha: float = 0.5,
):
    """
    Returns:
        macro      : mean over cells of cell_mean
        robust     : mean over cells of cell_min
        eval_score : mean over cells of ((1-alpha)*cell_mean + alpha*cell_min)
        il_mean    : global mean IL over cells
        ao_mean    : global mean AO over cells
        ol_mean    : global mean OL over cells
    """
    rng = np.random.default_rng(seed)

    futures = []
    meta = []
    for p_lo in p_lo_grid:
        for mode_name, cfg in teacher_modes:
            for _ in range(int(n_sessions_per_cell)):
                probs = sample_probs_this_session(session_K, float(p_hi), float(p_lo), rng)
                w = workers[len(futures) % len(workers)]
                futures.append(
                    w.run_session.remote(
                        agent_state_cpu,
                        probs,
                        print_this_session=False,
                        teacher_cfg=cfg,
                        return_obs=False,
                    )
                )
                meta.append((float(p_lo), str(mode_name)))

    results = _ray_get_in_batches(futures, batch_size=max(len(workers) * 4, 32))

    il_scores = {}
    ao_scores = {}
    ol_scores = {}

    for (p_lo, mode_name), res in zip(meta, results):
        (
            _, _, _,
            _, _, _, _,
            _, _,
            _,
            cum_rew_il, cum_rew_ao, cum_rew_ol,
            _, _, _,
            *_
        ) = res

        cell = (p_lo, mode_name)
        il_scores.setdefault(cell, []).append(float(cum_rew_il))
        ao_scores.setdefault(cell, []).append(float(cum_rew_ao))
        ol_scores.setdefault(cell, []).append(float(cum_rew_ol))

    # global condition means for debug
    il_mean = float(np.mean([np.mean(v) for v in il_scores.values()])) if il_scores else 0.0
    ao_mean = float(np.mean([np.mean(v) for v in ao_scores.values()])) if ao_scores else 0.0
    ol_mean = float(np.mean([np.mean(v) for v in ol_scores.values()])) if ol_scores else 0.0

    cell_means = []
    cell_mins = []
    cell_scores = []

    for cell in il_scores.keys():
        vals = np.array([
            np.mean(il_scores[cell]),
            np.mean(ao_scores[cell]),
            np.mean(ol_scores[cell]),
        ], dtype=np.float32)

        cell_mean = float(vals.mean())
        cell_min = float(vals.min())
        score = float((1.0 - alpha) * cell_mean + alpha * cell_min)

        cell_means.append(cell_mean)
        cell_mins.append(cell_min)
        cell_scores.append(score)

    macro = float(np.mean(cell_means)) if cell_means else 0.0
    robust = float(np.mean(cell_mins)) if cell_mins else 0.0
    eval_score = float(np.mean(cell_scores)) if cell_scores else 0.0

    return macro, robust, eval_score, il_mean, ao_mean, ol_mean

# -----------------------------
# Post-training grid eval (teacher modes × reward probs)
# -----------------------------

def eval_post_grid(
    workers,
    agent_state_cpu,
    session_K: int,
    p_hi: float,
    p_lo_grid: List[float],
    teacher_modes: List[Tuple[str, Dict[str, Any]]],
    n_sessions_per_cell: int,
    seed: int,
):
    """Evaluate a grid and return per-cell means + std.

    Returns:
      cell_stats[(p_lo, mode_name)] = {
        'pct_hi_il_mean', 'pct_hi_il_std', ...
        'cum_rew_il_mean', ...
        'pertrial_il_mean': (session_N,), ...
      }
    """
    rng = np.random.default_rng(seed)

    futures = []
    meta = []  # (p_lo, mode_name)
    for p_lo in p_lo_grid:
        for mode_name, cfg in teacher_modes:
            for _ in range(int(n_sessions_per_cell)):
                probs = sample_probs_this_session(session_K, float(p_hi), float(p_lo), rng)
                w = workers[len(futures) % len(workers)]
                futures.append(
                    w.run_session.remote(
                        agent_state_cpu,
                        probs,
                        print_this_session=False,
                        teacher_cfg=cfg,
                        return_obs=False,
                    )
                )
                meta.append((float(p_lo), str(mode_name)))

    results = _ray_get_in_batches(futures, batch_size=max(len(workers) * 4, 32))

    # collect per-cell arrays
    per_cell: Dict[Tuple[float, str], Dict[str, List[Any]]] = {}
    for (p_lo, mode_name), res in zip(meta, results):
        (
            _, _, _,
            _, _, _, _,
            _, _,
            _,
            cum_rew_il, cum_rew_ao, cum_rew_ol,
            pct_hi_il, pct_hi_ao, pct_hi_ol,
            high_reward_choices_il,
            high_reward_choices_ao,
            high_reward_choices_ol,
            _, _,  # logp_old_buf, v_old_buf
            g_trace_ao, g_trace_ol,
        ) = res

        cell = (p_lo, mode_name)
        d = per_cell.setdefault(cell, {
            "pct_hi_il": [], "pct_hi_ao": [], "pct_hi_ol": [],
            "cum_rew_il": [], "cum_rew_ao": [], "cum_rew_ol": [],
            "pertrial_il": [], "pertrial_ao": [], "pertrial_ol": [],
            "gtrace_ao": [], "gtrace_ol": [],
            "ao_rel": [], "ol_rel": [],   # per-session reliability target traces
        })
        d["pct_hi_il"].append(float(pct_hi_il))
        d["pct_hi_ao"].append(float(pct_hi_ao))
        d["pct_hi_ol"].append(float(pct_hi_ol))
        d["cum_rew_il"].append(float(cum_rew_il))
        d["cum_rew_ao"].append(float(cum_rew_ao))
        d["cum_rew_ol"].append(float(cum_rew_ol))
        d["pertrial_il"].append(np.asarray(high_reward_choices_il, dtype=np.float32))
        d["pertrial_ao"].append(np.asarray(high_reward_choices_ao, dtype=np.float32))
        d["pertrial_ol"].append(np.asarray(high_reward_choices_ol, dtype=np.float32))
        d["gtrace_ao"].append(np.asarray(g_trace_ao, dtype=np.float32))
        d["gtrace_ol"].append(np.asarray(g_trace_ol, dtype=np.float32))

    cell_stats: Dict[Tuple[float, str], Dict[str, Any]] = {}
    for cell, d in per_cell.items():
        def _mstd(x):
            x = np.asarray(x, dtype=np.float32)
            return float(x.mean()), float(x.std(ddof=0))

        n_sess = len(d["pct_hi_il"])

        il_m, il_s = _mstd(d["pct_hi_il"])
        ao_m, ao_s = _mstd(d["pct_hi_ao"])
        ol_m, ol_s = _mstd(d["pct_hi_ol"])
        ril_m, ril_s = _mstd(d["cum_rew_il"])
        rao_m, rao_s = _mstd(d["cum_rew_ao"])
        rol_m, rol_s = _mstd(d["cum_rew_ol"])

        # per-trial hit rates: (n_sessions, session_N) float32 with possible NaN at
        # unvisited positions → use nanmean/nanstd; convert std → SE for plotting
        pt_il = np.stack(d["pertrial_il"], axis=0)   # (S, T) in [0,1] or NaN
        pt_ao = np.stack(d["pertrial_ao"], axis=0)
        pt_ol = np.stack(d["pertrial_ol"], axis=0)
        pt_il_mean = np.nanmean(pt_il, axis=0)
        pt_ao_mean = np.nanmean(pt_ao, axis=0)
        pt_ol_mean = np.nanmean(pt_ol, axis=0)
        # per-position valid count (may differ if some positions were never reached)
        pt_il_n = np.sum(~np.isnan(pt_il), axis=0).clip(min=1)
        pt_ao_n = np.sum(~np.isnan(pt_ao), axis=0).clip(min=1)
        pt_ol_n = np.sum(~np.isnan(pt_ol), axis=0).clip(min=1)
        pt_il_se = np.nanstd(pt_il, axis=0, ddof=0) / np.sqrt(pt_il_n)
        pt_ao_se = np.nanstd(pt_ao, axis=0, ddof=0) / np.sqrt(pt_ao_n)
        pt_ol_se = np.nanstd(pt_ol, axis=0, ddof=0) / np.sqrt(pt_ol_n)

        # g traces: use nanmean/SE across sessions
        g_ao = np.stack(d["gtrace_ao"], axis=0)   # (S, T)
        g_ol = np.stack(d["gtrace_ol"], axis=0)
        gao_n   = np.sum(~np.isnan(g_ao), axis=0).clip(min=1)
        gol_n   = np.sum(~np.isnan(g_ol), axis=0).clip(min=1)
        gao_mu  = np.nanmean(g_ao, axis=0)
        gol_mu  = np.nanmean(g_ol, axis=0)
        gao_sd  = np.nanstd(g_ao,  axis=0, ddof=0)
        gol_sd  = np.nanstd(g_ol,  axis=0, ddof=0)
        gao_se  = gao_sd / np.sqrt(gao_n)   # uncertainty on the mean
        gol_se  = gol_sd / np.sqrt(gol_n)

        cell_stats[cell] = {
            "n_sessions": n_sess,
            "pct_hi_il_mean": il_m,  "pct_hi_il_std": il_s,
            "pct_hi_ao_mean": ao_m,  "pct_hi_ao_std": ao_s,
            "pct_hi_ol_mean": ol_m,  "pct_hi_ol_std": ol_s,
            "cum_rew_il_mean": ril_m, "cum_rew_il_std": ril_s,
            "cum_rew_ao_mean": rao_m, "cum_rew_ao_std": rao_s,
            "cum_rew_ol_mean": rol_m, "cum_rew_ol_std": rol_s,
            "pertrial_il_mean": pt_il_mean, "pertrial_il_se": pt_il_se,
            "pertrial_ao_mean": pt_ao_mean, "pertrial_ao_se": pt_ao_se,
            "pertrial_ol_mean": pt_ol_mean, "pertrial_ol_se": pt_ol_se,
            "gtrace_ao_mean": gao_mu, "gtrace_ao_se": gao_se, "gtrace_ao_std": gao_sd,
            "gtrace_ol_mean": gol_mu, "gtrace_ol_se": gol_se, "gtrace_ol_std": gol_sd,
            "gtrace_ao_mean_over_trials": float(np.nanmean(gao_mu)),
            "gtrace_ol_mean_over_trials": float(np.nanmean(gol_mu)),
        }

    return cell_stats


# -----------------------------
# Rollout
# -----------------------------

def meta_ep_rollout(env, agent, device, session_K: int, session_N: int, worker_id: int = 0, print_this_session: bool = False, return_obs: bool = True):
    """One full session rollout producing buffers used by update2.

    Returns a tuple:
      motor buffers: xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf
      bandit buffers: obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf, actor_buf, trial_cond_buf
      teacher_correct_buf: [-1 for non-observation trials, else 0/1 if teacher chose high reward]
      summary stats: cum_rewards_il/ao/ol, high_reward_choices_{il,ao,ol} (arrays over trials)
    """

    def log(*args, **kwargs):
        if print_this_session and worker_id == 0:
            print(*args, **kwargs)

    # Attention memory (past trials)
    k_tokens = []      # keys: q_in(chosen_feat)
    v_tokens = []      # values: action+actor+reward embeddings
    cond_tokens = []   # trial condition per past trial

    obs, info = env.reset()
    done = False

    pair_index_counter_il = np.ones(session_K, dtype=np.int32) * -1
    pair_index_counter_ao = np.ones(session_K, dtype=np.int32) * -1
    pair_index_counter_ol = np.ones(session_K, dtype=np.int32) * -1
    # numerator: high-reward choices at each within-pair position
    high_reward_choices_il = np.zeros(session_N, dtype=np.int32)
    high_reward_choice_ao  = np.zeros(session_N, dtype=np.int32)
    high_reward_choice_ol  = np.zeros(session_N, dtype=np.int32)
    # denominator: total choices (high + low) at each position — needed for a proper hit rate
    total_choices_il = np.zeros(session_N, dtype=np.int32)
    total_choices_ao = np.zeros(session_N, dtype=np.int32)
    total_choices_ol = np.zeros(session_N, dtype=np.int32)

    # scalar accuracy counters for self-choice trials
    hi_cnt_il = hi_cnt_ao = hi_cnt_ol = 0
    tot_il    = tot_ao    = tot_ol    = 0

    meta_ep_len = 0
    ep_start_flag = 1.0

    # bandit buffers (per trial)
    left_obs = [] if return_obs else None
    right_obs = [] if return_obs else None

    chosen_bandits_buf = []
    bandit_rewards_buf = []
    meta_ep_start_buf = []
    trial_cond_buf = []
    actor_buf = []
    teacher_correct_buf = []

    bandit_action_buf = []
    bandit_logp_buf   = []
    bandit_value_buf  = []
    bandit_policy_mask_buf = []  # 1 for IL/AO-s/OL-s, else 0

    # state for the current trial
    curr_trial_condition = None
    left_feats = None
    right_feats = None
    choice_target = None
    a_t = 0

    p_left = torch.tensor(0.5)
    p_right = torch.tensor(0.5)

    delta_val = np.array([0.0,0.0])
    g_val = 0.0

    # Track gate values (g) over within-pair trial index (0..session_N-1) for AO-s and OL-s.
    # We aggregate across pairs within a session by summing g at the corresponding within-pair index,
    # then averaging (counts) at the end of the session.
    g_trace_ao_sum = np.zeros(session_N, dtype=np.float32)
    g_trace_ao_cnt = np.zeros(session_N, dtype=np.int32)
    g_trace_ol_sum = np.zeros(session_N, dtype=np.float32)
    g_trace_ol_cnt = np.zeros(session_N, dtype=np.int32)

    def _ctx_for(q_tok: torch.Tensor, keep_conds: set) -> torch.Tensor:
        """Return ctx for a single query token, attending only to past trials in keep_conds.

        q_tok: (1,1,H) ; returns: (1,1,H)
        """
        if len(k_tokens) == 0:
            return agent.attn_ln(q_tok)

        k = torch.stack(k_tokens, dim=0)  # (Lpast,1,H)
        v = torch.stack(v_tokens, dim=0)  # (Lpast,1,H)
        keep = torch.as_tensor([c in keep_conds for c in cond_tokens], device=device, dtype=torch.bool)  # (Lpast,)
        kpm = (~keep).unsqueeze(0)  # (1,Lpast) True=mask
        ctx_attn = agent._safe_causal_attn(q_tok, k, v, kpm)  # (1,1,H)
        return agent.attn_ln(ctx_attn + q_tok)

    def _belief_policy_value(curr_cond: str, lf: torch.Tensor, rf: torch.Tensor):
        """Compute detached control policy/value and current belief summaries for one trial."""
        ql = agent.q_in(lf).unsqueeze(0)  # (1,1,H)
        qr = agent.q_in(rf).unsqueeze(0)  # (1,1,H)

        lf_seq = lf.unsqueeze(0)  # (1,1,F)
        rf_seq = rf.unsqueeze(0)  # (1,1,F)

        zeros_ctx_l = torch.zeros_like(ql)
        zeros_ctx_r = torch.zeros_like(qr)
        zeros_logits = torch.zeros((1, 1, 2), device=device, dtype=lf.dtype)
        zeros_gate = torch.zeros((1, 1, 1), device=device, dtype=lf.dtype)

        if curr_cond == "IL":
            ctx_l_cur = _ctx_for(ql, {"IL"})
            ctx_r_cur = _ctx_for(qr, {"IL"})
            ctx_l_obs = zeros_ctx_l
            ctx_r_obs = zeros_ctx_r

            q_self_logits = torch.cat([
                agent._head(agent.q_base, ctx_l_cur, lf_seq),
                agent._head(agent.q_base, ctx_r_cur, rf_seq),
            ], dim=-1)
            q_teacher_ao = zeros_logits
            q_teacher_ol = zeros_logits
            g_ao = zeros_gate
            g_ol = zeros_gate
            use_ao = zeros_gate
            use_ol = zeros_gate

        elif curr_cond == "AO-s":
            ctx_l_cur = _ctx_for(ql, {"AO-s"})
            ctx_r_cur = _ctx_for(qr, {"AO-s"})
            ctx_l_obs = _ctx_for(ql, {"AO-o"})
            ctx_r_obs = _ctx_for(qr, {"AO-o"})

            q_self_logits = torch.cat([
                agent._head(agent.q_base, ctx_l_cur, lf_seq),
                agent._head(agent.q_base, ctx_r_cur, rf_seq),
            ], dim=-1)
            q_teacher_ao = torch.cat([
                agent._head(agent.q_delta_ao, ctx_l_obs, lf_seq),
                agent._head(agent.q_delta_ao, ctx_r_obs, rf_seq),
            ], dim=-1)
            q_teacher_ol = zeros_logits
            g_ao = torch.sigmoid(agent.gate_ao(torch.cat([ctx_l_cur, ctx_r_cur, ctx_l_obs, ctx_r_obs], dim=-1))).clamp(0.0, 1.0)
            g_ol = zeros_gate
            use_ao = torch.full_like(g_ao, 1.0 if any(c == "AO-o" for c in cond_tokens) else 0.0)
            use_ol = zeros_gate

        elif curr_cond == "OL-s":
            ctx_l_cur = _ctx_for(ql, {"OL-s"})
            ctx_r_cur = _ctx_for(qr, {"OL-s"})
            ctx_l_obs = _ctx_for(ql, {"OL-o"})
            ctx_r_obs = _ctx_for(qr, {"OL-o"})

            q_self_logits = torch.cat([
                agent._head(agent.q_base, ctx_l_cur, lf_seq),
                agent._head(agent.q_base, ctx_r_cur, rf_seq),
            ], dim=-1)
            q_teacher_ao = zeros_logits
            q_teacher_ol = torch.cat([
                agent._head(agent.q_delta_ol, ctx_l_obs, lf_seq),
                agent._head(agent.q_delta_ol, ctx_r_obs, rf_seq),
            ], dim=-1)
            g_ao = zeros_gate
            g_ol = torch.sigmoid(agent.gate_ol(torch.cat([ctx_l_cur, ctx_r_cur, ctx_l_obs, ctx_r_obs], dim=-1))).clamp(0.0, 1.0)
            use_ao = zeros_gate
            use_ol = torch.full_like(g_ol, 1.0 if any(c == "OL-o" for c in cond_tokens) else 0.0)

        elif curr_cond == "AO-o":
            ctx_l_cur = zeros_ctx_l
            ctx_r_cur = zeros_ctx_r
            ctx_l_obs = _ctx_for(ql, {"AO-o"})
            ctx_r_obs = _ctx_for(qr, {"AO-o"})
            q_self_logits = zeros_logits
            q_teacher_ao = torch.cat([
                agent._head(agent.q_delta_ao, ctx_l_obs, lf_seq),
                agent._head(agent.q_delta_ao, ctx_r_obs, rf_seq),
            ], dim=-1)
            q_teacher_ol = zeros_logits
            g_ao = zeros_gate
            g_ol = zeros_gate
            use_ao = zeros_gate
            use_ol = zeros_gate

        elif curr_cond == "OL-o":
            ctx_l_cur = zeros_ctx_l
            ctx_r_cur = zeros_ctx_r
            ctx_l_obs = _ctx_for(ql, {"OL-o"})
            ctx_r_obs = _ctx_for(qr, {"OL-o"})
            q_self_logits = zeros_logits
            q_teacher_ao = zeros_logits
            q_teacher_ol = torch.cat([
                agent._head(agent.q_delta_ol, ctx_l_obs, lf_seq),
                agent._head(agent.q_delta_ol, ctx_r_obs, rf_seq),
            ], dim=-1)
            g_ao = zeros_gate
            g_ol = zeros_gate
            use_ao = zeros_gate
            use_ol = zeros_gate

        else:
            ctx_l_cur = _ctx_for(ql, {"IL"})
            ctx_r_cur = _ctx_for(qr, {"IL"})
            ctx_l_obs = zeros_ctx_l
            ctx_r_obs = zeros_ctx_r
            q_self_logits = torch.cat([
                agent._head(agent.q_base, ctx_l_cur, lf_seq),
                agent._head(agent.q_base, ctx_r_cur, rf_seq),
            ], dim=-1)
            q_teacher_ao = zeros_logits
            q_teacher_ol = zeros_logits
            g_ao = zeros_gate
            g_ol = zeros_gate
            use_ao = zeros_gate
            use_ol = zeros_gate

        p_self = torch.sigmoid(q_self_logits)
        p_teacher_ao = torch.sigmoid(q_teacher_ao)
        p_teacher_ol = torch.sigmoid(q_teacher_ol)

        p_belief = p_self
        if curr_cond == "AO-s":
            p_belief = torch.where(use_ao.bool(), (1.0 - g_ao) * p_belief + g_ao * p_teacher_ao, p_belief)
        elif curr_cond == "OL-s":
            p_belief = torch.where(use_ol.bool(), (1.0 - g_ol) * p_belief + g_ol * p_teacher_ol, p_belief)

        control_in = agent._control_features(
            ctx_l_cur.detach(),
            ctx_r_cur.detach(),
            ctx_l_obs.detach(),
            ctx_r_obs.detach(),
            lf_seq.detach(),
            rf_seq.detach(),
            p_self.detach(),
            p_teacher_ao.detach(),
            p_teacher_ol.detach(),
            p_belief.detach(),
            g_ao.detach(),
            g_ol.detach(),
            use_ao.detach(),
            use_ol.detach(),
        )

        policy_logits = agent.pi_head(control_in)
        v = agent.v_head(control_in)
        belief_delta = p_belief - p_self
        gate_out = g_ao if curr_cond == "AO-s" else (g_ol if curr_cond == "OL-s" else None)
        return policy_logits, v, belief_delta, gate_out

    def _choice_logits(curr_cond: str, lf: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
        logits, _, belief_delta, gate_out = _belief_policy_value(curr_cond, lf, rf)
        return logits, belief_delta, gate_out

    def _value_pred(curr_cond: str, lf: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
        _, v, _, _ = _belief_policy_value(curr_cond, lf, rf)
        return v

    while not done:
        # CHOICE at trial start — obs_tensor is only needed here (for CNN encoding).
        # Skipping it on every motor step eliminates ~8 000 large tensor ops per session.
        if ep_start_flag == 1.0:
            obs_tensor = (
                torch.as_tensor(obs, device=device)
                .permute(2, 0, 1).unsqueeze(0)
                .to(torch.float32).div_(255.0)
            )
            left_view, right_view = extract_lr_views(obs_tensor, env, crop_size=112, pad=6)
            curr_trial_condition = info.get("curr_trial_condition")

            left_feats = agent.encode(left_view)    # (1,F)
            right_feats = agent.encode(right_view)  # (1,F)

            if return_obs:
                # Store uint8 (0-255) instead of float32 (0-1) — 4× smaller Ray transfer.
                # update2 (learner) decodes back to float32 on the GPU.
                left_obs.append(left_view.squeeze(0).mul(255.0).clamp_(0, 255).byte().cpu().numpy())
                right_obs.append(right_view.squeeze(0).mul(255.0).clamp_(0, 255).byte().cpu().numpy())

            # Self-choice trials
            if curr_trial_condition not in ("AO-o", "OL-o"):
                logits, delta, g = _choice_logits(curr_trial_condition, left_feats, right_feats)  # (1,1,2)
                delta_val = np.array(delta[0,0].cpu().numpy()) if delta is not None else np.array([-1,-1])
                g_val = g[0,0].item() if g is not None else -1
                # print("delta.shape = ", delta.shape, ", g.shape = ", g.shape, g_val, delta_val)
                # tau = 0.2
                # p_choose = torch.softmax(logits / tau, dim=-1)  # (1,1,2)
                # p_left = p_choose[0, 0, 0]
                # p_right = p_choose[0, 0, 1]
                # dist = torch.distributions.Categorical(probs=p_choose[0, 0])
                # a_t = int(dist.sample().item())  # 0=left, 1=right


                tau = 1.0  # let entropy bonus drive exploration; you can anneal later
                dist = torch.distributions.Categorical(logits=(logits[0, 0] / tau))
                p_left = torch.softmax(logits[0, 0], dim=-1)[0]
                p_right = torch.softmax(logits[0, 0], dim=-1)[1]
                a_t = int(dist.sample().item())
                logp_old = float(dist.log_prob(torch.tensor(a_t, device=device)).item())
                v_old = float(_value_pred(curr_trial_condition, left_feats, right_feats).item())
                bandit_action_buf.append(a_t)
                bandit_logp_buf.append(logp_old)
                bandit_value_buf.append(v_old)
                bandit_policy_mask_buf.append(1.0)

                choice_target = F.one_hot(torch.tensor(a_t, device=device), num_classes=2).unsqueeze(0).float()
            else:
                # Observational trial: env/teacher chooses (revealed at trial end).
                a_t = 0  # dummy; replaced at trial end
                choice_target = F.one_hot(torch.tensor(a_t, device=device), num_classes=2).unsqueeze(0).float()

                v_old = float(_value_pred(curr_trial_condition, left_feats, right_feats).item())
                bandit_action_buf.append(0)     # dummy
                bandit_logp_buf.append(0.0)     # dummy
                bandit_value_buf.append(v_old)
                bandit_policy_mask_buf.append(0.0)


            meta_ep_len += 1
            ep_start_flag = 0.0

        # ── DIRECT CHOICE: teleport cursor to target instead of running motor policy ──
        # Self-choice trials (IL, AO-s, OL-s): cursor is set to agent's chosen bandit center.
        #   → env registers the selection on the very next step (1 step per trial instead of ~100).
        # Obs trials (AO-o, OL-o): try to discover teacher's side from env attributes so we
        #   can teleport there too.  If the attribute is not found the cursor stays put and
        #   the env advances the teacher's cursor internally each step until trial_ended fires.
        if ep_start_flag == 0.0:   # just made a choice above
            uw = env.unwrapped
            left_c  = np.array([uw.left_rect.centerx,  uw.left_rect.centery], np.float32)
            right_c = np.array([uw.right_rect.centerx, uw.right_rect.centery], np.float32)

            if curr_trial_condition not in ("AO-o", "OL-o"):
                # Direct teleport to agent's chosen bandit
                uw.cursor = left_c if a_t == 0 else right_c
            else:
                # Attempt to read teacher's chosen side from env (attribute name varies by
                # env version). If found, teleport there so obs trials also end in 1 step.
                for _attr in ('teacher_side', 'teacher_chosen_side', 'teacher_target_side',
                              'obs_target_side', 'selected_target_this_trial'):
                    _ts = getattr(uw, _attr, None)
                    if _ts is not None:
                        uw.cursor = left_c if int(_ts) == 0 else right_c
                        break

        # Poll env until this trial ends.  Use a zero motor action — cursor is already
        # at target for self-choice trials so the env registers arrival on the first call.
        _dummy = np.zeros(2, dtype=np.float32)
        while True:
            next_obs, reward, term, _, info = env.step(_dummy)
            obs   = next_obs
            done  = term
            if info.get("trial_ended") or done:
                break

        # Trial ended => append memory token
        if info.get("trial_ended"):
            meta_ep_start = 0.0 if meta_ep_len > 1 else 1.0

            actor = 0
            if curr_trial_condition in ("AO-o", "OL-o"):
                # on obs trials, use env-chosen action
                a_t = int(info.get("selected_target", 0))
                choice_target = F.one_hot(torch.tensor(a_t, device=device), num_classes=2).unsqueeze(0).float()
                actor = 1

            # reward index used by the model: 0/1 observed, 2 = no_feedback
            if curr_trial_condition == "AO-o":
                rwd_idx = 2
            else:
                rwd_idx = int(round(float(reward)))
                rwd_idx = 1 if rwd_idx >= 1 else 0

            # key/value tokens (match update2)
            aL = choice_target[..., 0:1]
            aR = choice_target[..., 1:2]
            chosen_feat = aL * left_feats + aR * right_feats

            action_emb = agent.action(torch.tensor([a_t], device=device, dtype=torch.long))     # (1,H)
            actor_emb  = agent.actor(torch.tensor([actor], device=device, dtype=torch.long))    # (1,H)
            rwd_emb    = agent.rwd_in(torch.tensor([rwd_idx], device=device, dtype=torch.long)) # (1,H)
            v_tok = action_emb + actor_emb + rwd_emb  # (1,H)

            k_tok = agent.q_in(chosen_feat)  # (1,H)

            v_tokens.append(v_tok)
            k_tokens.append(k_tok)
            cond_tokens.append(curr_trial_condition)

            # perf stats
            pair_index_ep = info.get("prev_pair_index_in_session", -1)
            selected_high_reward = info.get("selected_high_reward_this_trial", -1)
            sh = int(selected_high_reward) if selected_high_reward is not None else -1
            flipped = info.get("side_is_flipped", False)

            if curr_trial_condition == "IL":
                pair_index_counter_il[pair_index_ep] += 1
            elif curr_trial_condition == "AO-s":
                pair_index_counter_ao[pair_index_ep] += 1
            elif curr_trial_condition == "OL-s":
                pair_index_counter_ol[pair_index_ep] += 1

            # Gate traces (g) for AO-s / OL-s at the within-pair trial index.
            # NOTE: g_val is computed at trial start for self-choice trials; we snapshot it here when the trial ends.
            if (pair_index_ep is not None) and (pair_index_ep >= 0):
                if curr_trial_condition == "AO-s":
                    idx = int(pair_index_counter_ao[pair_index_ep])
                    if 0 <= idx < session_N and np.isfinite(float(g_val)):
                        g_trace_ao_sum[idx] += float(g_val)
                        g_trace_ao_cnt[idx] += 1
                elif curr_trial_condition == "OL-s":
                    idx = int(pair_index_counter_ol[pair_index_ep])
                    if 0 <= idx < session_N and np.isfinite(float(g_val)):
                        g_trace_ol_sum[idx] += float(g_val)
                        g_trace_ol_cnt[idx] += 1

            if curr_trial_condition == "IL":
                tot_il += 1
                if sh == 1:
                    hi_cnt_il += 1
            elif curr_trial_condition == "AO-s":
                tot_ao += 1
                if sh == 1:
                    hi_cnt_ao += 1
            elif curr_trial_condition == "OL-s":
                tot_ol += 1
                if sh == 1:
                    hi_cnt_ol += 1

            # legacy per-index counters (per-trial plot)
            if curr_trial_condition == "IL":
                pos = pair_index_counter_il[pair_index_ep]
                total_choices_il[pos] += 1
                if sh == 1:
                    high_reward_choices_il[pos] += 1
            elif curr_trial_condition == "AO-s":
                pos = pair_index_counter_ao[pair_index_ep]
                total_choices_ao[pos] += 1
                if sh == 1:
                    high_reward_choice_ao[pos] += 1
            elif curr_trial_condition == "OL-s":
                pos = pair_index_counter_ol[pair_index_ep]
                total_choices_ol[pos] += 1
                if sh == 1:
                    high_reward_choice_ol[pos] += 1


            pair_probs = env.unwrapped.pair_probs
            log(
                "Ep:", info.get("trial_index"),
                "| curr_idx =", pair_index_ep,
                "| iter_il =", pair_index_counter_il[pair_index_ep],
                "| iter_ao =", pair_index_counter_ao[pair_index_ep],
                "| iter_ol =", pair_index_counter_ol[pair_index_ep],
                "| curr_trial_condition =", curr_trial_condition,
                "| choice target:", int(choice_target.argmax(dim=-1).item()),
                "| Reached Target:", info.get("selected_target"),
                "| reward_idx:", rwd_idx,
                "| flipped =", flipped,
                "| selected_high_reward =", selected_high_reward,
                "| p_left =", round(float(p_left), 2),
                "| p_right =", round(float(p_right), 2),
                "| v_old =", round(float(v_old), 2),
                "| delta = ", np.round(delta_val, 2),
                "| g =", np.round(float(g_val), 2),
                "| pair_probs =", [(round(pl, 2), round(pr, 2)) for pl, pr in pair_probs]
            )

            chosen_bandits_buf.append(choice_target.squeeze(0).detach().cpu().numpy())
            bandit_rewards_buf.append(float(rwd_idx))  # store idx so update2 can mask AO-o with 2
            meta_ep_start_buf.append(float(meta_ep_start))
            actor_buf.append(actor)
            trial_cond_buf.append(curr_trial_condition)

            # teacher correctness label (optional supervision)
            tc = -1.0
            if curr_trial_condition in ("AO-o", "OL-o"):
                if selected_high_reward in (-1, None):
                    tc = -1.0
                else:
                    tc = 1.0 if bool(selected_high_reward) else 0.0
            teacher_correct_buf.append(float(tc))

            # next trial
            choice_target = None
            ep_start_flag = 1.0

    bandit_rewards_arr = np.asarray(bandit_rewards_buf, dtype=np.float32)
    trial_cond_arr = np.asarray(trial_cond_buf)

    mask_il = (trial_cond_arr == "IL")
    mask_ao = (trial_cond_arr == "AO-s")
    mask_ol = (trial_cond_arr == "OL-s")

    mean_r_il = float(bandit_rewards_arr[mask_il].mean()) if mask_il.any() else 0.0
    mean_r_ao = float(bandit_rewards_arr[mask_ao].mean()) if mask_ao.any() else 0.0
    mean_r_ol = float(bandit_rewards_arr[mask_ol].mean()) if mask_ol.any() else 0.0

    cum_rewards_il = mean_r_il * float(session_N)   # session_N=12 → range ~[0,12]
    cum_rewards_ao = mean_r_ao * float(session_N)
    cum_rewards_ol = mean_r_ol * float(session_N)

    pct_hi_il = 100.0 * float(hi_cnt_il) / float(max(tot_il, 1))
    pct_hi_ao = 100.0 * float(hi_cnt_ao) / float(max(tot_ao, 1))
    pct_hi_ol = 100.0 * float(hi_cnt_ol) / float(max(tot_ol, 1))

    high_reward_choices_il = np.array(high_reward_choices_il, dtype=np.int32)
    high_reward_choices_ao = np.array(high_reward_choice_ao,  dtype=np.int32)
    high_reward_choices_ol = np.array(high_reward_choice_ol,  dtype=np.int32)

    # Convert to proper hit-rate in [0,1] using the actual denominator at each position.
    # Positions never visited (total_choices == 0) get NaN so they don't pollute the mean.
    def _hit_rate(num, den):
        den = np.asarray(den, dtype=np.float32)
        return np.where(den > 0, num.astype(np.float32) / den, np.nan)

    pertrial_rate_il = _hit_rate(high_reward_choices_il, total_choices_il)
    pertrial_rate_ao = _hit_rate(high_reward_choices_ao, total_choices_ao)
    pertrial_rate_ol = _hit_rate(high_reward_choices_ol, total_choices_ol)

    # Mean gate value per within-pair trial index (NaN where not observed in this session)
    g_trace_ao = np.where(g_trace_ao_cnt > 0, g_trace_ao_sum / np.maximum(g_trace_ao_cnt, 1), np.nan).astype(np.float32)
    g_trace_ol = np.where(g_trace_ol_cnt > 0, g_trace_ol_sum / np.maximum(g_trace_ol_cnt, 1), np.nan).astype(np.float32)


    if return_obs:
        obs_bandit = {
            "left":  np.stack(left_obs, axis=0),
            "right": np.stack(right_obs, axis=0),
        }
    else:
        obs_bandit = None


    return (
        [], [], [],   # motor buffers removed (direct-choice rollout)
        obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf,
        actor_buf, trial_cond_buf,
        teacher_correct_buf,
        cum_rewards_il, cum_rewards_ao, cum_rewards_ol,
        pct_hi_il, pct_hi_ao, pct_hi_ol,
        pertrial_rate_il, pertrial_rate_ao, pertrial_rate_ol,   # proper [0,1] hit rates
        bandit_logp_buf, bandit_value_buf,
        g_trace_ao, g_trace_ol
    )


# -----------------------------
# Ray worker
# -----------------------------

@ray.remote(num_cpus=1, num_gpus=0)
class RolloutWorker:
    def __init__(self, session_K: int, session_N: int, seed: int = 0, worker_id: int = 0):
        self.session_K = session_K
        self.session_N = session_N
        self.device = torch.device("cpu")
        self.worker_id = worker_id

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

        # Snapshot teacher-related defaults once, so each session can start from a clean slate.
        uw = self.env.unwrapped
        self._teacher_defaults = {}
        for k in TEACHER_CFG_KEYS:
            if hasattr(uw, k):
                try:
                    self._teacher_defaults[k] = copy.deepcopy(getattr(uw, k))
                except Exception:
                    # Fallback: store raw value
                    self._teacher_defaults[k] = getattr(uw, k)

        # Build the agent once and reuse it across sessions (load_state_dict only).
        _hidden = 128
        _fdim   = 128
        self._agent = bl.BanditLearner(
            input_size=_fdim + 1,
            feature_dim=_fdim,
            rnn_hidden_size=_hidden,
            action_dim=2,
            num_pairs=session_K,
            max_trials=session_N * session_K,
        ).to(self.device)
        self._agent.eval()

        # Limit intra-op threads so workers don't fight over CPU cores.
        torch.set_num_threads(1)

    def run_session(self, agent_state_dict, probs_this_session, print_this_session: bool = False, teacher_cfg: dict = None, return_obs: bool = True):
        """Run one session with given pair probs. teacher_eps is optional (only used if env supports it)."""
        # Reuse the pre-built agent — just hot-swap weights (much cheaper than full construction).
        self._agent.load_state_dict(agent_state_dict)
        self._agent.eval()
        agent = self._agent

        self.env.unwrapped.pair_probs = probs_this_session

        # Reset teacher-related attributes to defaults, then apply teacher_cfg.
        # This avoids leakage when we run mixed teacher modes back-to-back in the same worker.
        uw = self.env.unwrapped
        for k, v in self._teacher_defaults.items():
            if hasattr(uw, k):
                try:
                    setattr(uw, k, copy.deepcopy(v))
                except Exception:
                    try:
                        setattr(uw, k, v)
                    except Exception:
                        pass

        # Optional teacher configuration (expert/slow/fast/unreliable, eps/alpha/tau etc.)
        # We set only attributes that exist in the env to keep this backward-compatible.
        if teacher_cfg is not None:
            for k, v in dict(teacher_cfg).items():
                if hasattr(uw, k):
                    try:
                        setattr(uw, k, copy.deepcopy(v))
                    except Exception:
                        try:
                            setattr(uw, k, v)
                        except Exception:
                            pass

        # print(f"Worker {self.worker_id} running session with probs {probs_this_session} and teacher_cfg {teacher_cfg}")                                
        with torch.no_grad():
            return meta_ep_rollout(
                self.env, agent, self.device,
                self.session_K, self.session_N,
                worker_id=self.worker_id,
                print_this_session=print_this_session,
                return_obs=return_obs
            )


# -----------------------------
# Main
# -----------------------------

# bandit_train_batch.py

def _call_update2(agent, optim_bandit, optim_motor,
                 batch_xy_pos, batch_goal_vec, batch_chosen_bandits_motor,
                 batch_bandit_obs, batch_chosen_bandits, batch_bandit_rewards, batch_meta_ep_start,
                 batch_actor, batch_trial_cond, batch_teacher_correct,
                 batch_logp_old, batch_v_old, device,
                 ppo_epochs: int = 4, ppo_minibatch_size: int = 256,
                 aux_mb_size: int = 256,
                 self_bce_coef: float = 1.0, aux_ao_coef: float = 0.2, aux_ol_coef: float = 0.5,
                 gate_ao_coef: float = 0.5, gate_ol_coef: float = 0.5, gate_ol_sg: bool = True,
                 session_chunk_size: int = 10):
    return agent.update2(
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
        batch_teacher_correct,
        batch_logp_old,
        batch_v_old,
        device,
        ppo_epochs=ppo_epochs,
        ppo_minibatch_size=ppo_minibatch_size,
        aux_mb_size=aux_mb_size,
        self_bce_coef=self_bce_coef,
        aux_ao_coef=aux_ao_coef,
        aux_ol_coef=aux_ol_coef,
        gate_ao_coef=gate_ao_coef,
        gate_ol_coef=gate_ol_coef,
        gate_ol_sg=gate_ol_sg,
        session_chunk_size=session_chunk_size,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=0)
    # FIX 5: Each Ray worker holds a live pygame env + BanditLearner in RAM for the
    # entire run.  40 workers can consume several GB before any rollout results arrive.
    # Prefer 16–24 workers; throughput barely changes because sessions run sequentially
    # within each worker.
    parser.add_argument('--num-workers', type=int, default=4)
    # If you want *balanced* training over the full 5×4 grid each update:
    #   episodes_per_update = grid_repeats * 20
    parser.add_argument('--grid-repeats', type=int, default=1,
                        help='K repeats of the full num_grid_cells-cell (p_lo × teacher_mode) grid per update. B = K*num_grid_cells')
    parser.add_argument('--episodes-per-update', type=int, default=num_grid_cells,
                        help='Rollout sessions per update (B). If --grid-repeats is set, B will be overridden to K*num_grid_cells.')
    parser.add_argument('--num-updates', type=int, default=1500)

    # PPO inner-loop
    parser.add_argument('--ppo-epochs', type=int, default=4)
    parser.add_argument('--ppo-minibatch-size', type=int, default=256)

    # Aux losses (supervised) for observation trials
    parser.add_argument('--aux-mb-size', type=int, default=256)
    parser.add_argument('--train-session-chunk-size', type=int, default=10,
                        help='Number of full sessions to move to GPU at once inside update2. Lower this if training still OOMs.')
    parser.add_argument('--self-bce-coef', type=float, default=1.0, help='Self-choice fused-belief BCE weight')
    parser.add_argument('--aux-ao-coef', type=float, default=0.2, help='AO-o teacher action imitation weight')
    parser.add_argument('--aux-ol-coef', type=float, default=0.5, help='OL-o teacher reward prediction weight')

    # Gate reliability losses  L_gate,ao = BCE(g_ao, ao_rel_target)
    #                           L_gate,ol = BCE(sg(g_ol), ol_rel_target)
    parser.add_argument('--gate-ao-coef', type=float, default=0.5,
                        help='Weight for AO gate reliability BCE loss (g_ao vs running AO-o teacher accuracy).')
    parser.add_argument('--gate-ol-coef', type=float, default=0.5,
                        help='Weight for OL gate reliability BCE loss (g_ol vs running OL-o teacher accuracy).')
    parser.add_argument('--no-gate-ol-sg', dest='gate_ol_sg', action='store_false', default=True,
                        help='Disable stop-gradient on g_ol in the OL gate loss (default: sg enabled, matching ↓ in formula).')

    # Grid eval + early stop (eval-based, not train EMA)
    parser.add_argument('--eval-interval', type=int, default=20)
    parser.add_argument('--eval-sessions-per-cell', type=int, default=5)
    parser.add_argument('--warmup-updates', type=int, default=400)
    parser.add_argument('--patience-evals', type=int, default=10)
    parser.add_argument('--min-delta', type=float, default=0.5)

    # Post-training eval / plots
    parser.add_argument('--eval-sessions', type=int, default=200)

    # Post-training *grid* eval: teacher modes × reward probs
    parser.add_argument('--post-grid-eval', dest='post_grid_eval', action='store_true', default=True,
                        help='Run post-training eval on a grid of (teacher_mode × p_lo) and save plots/CSV.')
    parser.add_argument('--no-post-grid-eval', dest='post_grid_eval', action='store_false',
                        help='Disable post-training grid eval.')
    parser.add_argument('--post-grid-sessions-per-cell', type=int, default=50,
                        help='Eval sessions per grid cell (teacher_mode × p_lo).')

    # Backward-compat alias
    parser.add_argument('--gap-sweep', dest='post_grid_eval', action='store_true',
                        help='(deprecated) Alias for --post-grid-eval.')
    parser.add_argument('--p-hi', type=float, default=0.8)
    parser.add_argument('--p-lo-min', type=float, default=0.10)
    parser.add_argument('--p-lo-max', type=float, default=0.6)
    parser.add_argument('--p-lo-steps', type=int, default=14)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # TensorBoard writer
    log_dir = os.path.join("runs", f"bandit_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    writer = SummaryWriter(log_dir)

    session_K = 3
    session_N = 30

    hidden_size = 128
    feature_dim = 128
    input_size = feature_dim + 1

    # eval-based early stopping (grid eval)
    best_eval = -float("inf")
    bad_evals = 0

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

    plot_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    run_dir = os.path.join(plot_dir, datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)


    # Ray init
    if not ray.is_initialized():
        tmp_dir = os.path.expanduser("~/tmp/ray")
        os.makedirs(tmp_dir, exist_ok=True)

        # FIX 1: Hard-cap the Ray shared-memory object store.
        # Without a limit Ray can consume most of RAM for object storage, leaving
        # workers vulnerable to the OOM killer when many large result objects pile up.
        # Tune this value to ~25–33% of your node's total RAM.
        ray.init(_temp_dir=tmp_dir, include_dashboard=False, ignore_reinit_error=True,
                 object_store_memory=6 * 1024 ** 3)  # 6 GB cap

    # rollout workers
    workers = [
        RolloutWorker.remote(session_K, session_N, seed=args.seed + 1000 * i, worker_id=i)
        for i in range(args.num_workers)
    ]
    num_updates = args.num_updates

    # Balanced grid training: always use B = K*20 sessions per update.
    # (Keeps coverage of all p_lo × teacher_mode combinations each update.)
    if args.grid_repeats is not None:
        if args.grid_repeats < 1:
            raise ValueError('--grid-repeats must be >= 1')
        B = int(args.grid_repeats) * num_grid_cells
        if args.episodes_per_update is not None and int(args.episodes_per_update) != B:
            print(f'[info] Overriding --episodes-per-update={args.episodes_per_update} -> {B} because --grid-repeats={args.grid_repeats}')
    else:
        B = int(args.episodes_per_update)
        if B % num_grid_cells != 0:
            raise ValueError(f'episodes-per-update must be a multiple of {num_grid_cells} ({len(P_LO_GRID_DEFAULT)} p_lo × {len(TEACHER_MODES_DEFAULT)} teacher modes).')

    W = args.num_workers

    # training mixture RNG
    rng_train = np.random.default_rng(args.seed + 123)

    # -----------------------------
    # Balanced training grid (20 cells)
    # -----------------------------
    train_cells = [(float(p_lo), mode_name, cfg)
                   for p_lo in P_LO_GRID_DEFAULT
                   for (mode_name, cfg) in TEACHER_MODES_DEFAULT]
    assert len(train_cells) == num_grid_cells, f'Expected {num_grid_cells} grid cells, got {len(train_cells)}'

    # -----------------------------------------------------------------
    # Helper: build shuffled cell list and dispatch all B rollout futures
    # at once using a single ray.put state-dict reference (avoids the
    # original W×serialise overhead and the mid-batch blocking barrier).
    # -----------------------------------------------------------------
    def _build_cell_list():
        cells = train_cells * int(args.grid_repeats)
        rng_train.shuffle(cells)
        return cells

    def _dispatch_rollouts(cells, agent_state_ref, update_idx):
        """Fire all B run_session tasks without waiting; returns future list."""
        futures = []
        for i, (p_lo, mode_name, cfg) in enumerate(cells):
            probs = sample_probs_this_session(
                session_K, p_hi=float(args.p_hi), p_lo=float(p_lo), rng=rng_train)
            print_flag = (i == 0) and (update_idx % 10 == 0)
            futures.append(
                workers[i % W].run_session.remote(
                    agent_state_ref,
                    probs,
                    print_this_session=print_flag,
                    teacher_cfg=cfg,
                )
            )
        return futures

    def _collect_rollouts(rollout_results):
        """Unpack ray results into training batch lists."""
        bxy, bgv, bcbm = [], [], []
        bobs, bcb, bbr, bmes = [], [], [], []
        bac, btc, btcor = [], [], []
        blp, bvold = [], []
        cr_il, cr_ao, cr_ol = [], [], []
        hrc_il, hrc_ao, hrc_ol = [], [], []

        for res in rollout_results:
            (
                xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
                obs_bandit, chosen_bandits_buf, bandit_rewards_buf, meta_ep_start_buf,
                actor_buf, trial_cond_buf,
                teacher_correct_buf,
                cum_rewards_il, cum_rewards_ao, cum_rewards_ol,
                pct_hi_il, pct_hi_ao, pct_hi_ol,
                high_reward_choices_il, high_reward_choices_ao, high_reward_choices_ol,
                logp_old_buf, v_old_buf, _, _
            ) = res

            bxy.extend(xy_pos_buf)
            bgv.extend(goal_vec_buf)
            bcbm.extend(chosen_bandits_motor_buf)
            bobs.append(obs_bandit)
            bcb.append(torch.as_tensor(np.stack(chosen_bandits_buf), dtype=torch.float32))
            bbr.append(torch.as_tensor(np.stack(bandit_rewards_buf), dtype=torch.float32))
            bmes.append(torch.as_tensor(np.stack(meta_ep_start_buf), dtype=torch.float32))
            bac.append(torch.as_tensor(np.stack(actor_buf), dtype=torch.long))
            btc.append(np.stack(trial_cond_buf))
            btcor.append(torch.as_tensor(np.stack(teacher_correct_buf), dtype=torch.float32))
            blp.append(torch.as_tensor(np.stack(logp_old_buf), dtype=torch.float32))
            bvold.append(torch.as_tensor(np.stack(v_old_buf), dtype=torch.float32))
            cr_il.append(cum_rewards_il)
            cr_ao.append(cum_rewards_ao)
            cr_ol.append(cum_rewards_ol)
            hrc_il.append(high_reward_choices_il)
            hrc_ao.append(high_reward_choices_ao)
            hrc_ol.append(high_reward_choices_ol)

        return (bxy, bgv, bcbm, bobs, bcb, bbr, bmes, bac, btc, btcor, blp, bvold,
                cr_il, cr_ao, cr_ol, hrc_il, hrc_ao, hrc_ol)

    # -----------------------------------------------------------------
    # Training loop: dispatch → collect (batched) → update → eval.
    #
    # FIX 4: Dispatch is now at the TOP of each update iteration rather than
    # being pre-launched before eval.  The previous "overlap" design fired the
    # next training batch (320 futures) BEFORE eval ran, then eval fired up to
    # 400 more futures on top — both sets of results lived simultaneously in the
    # Ray object store, easily exceeding available RAM and triggering the OOM
    # killer.  Moving dispatch after eval eliminates that peak entirely.
    #
    # FIX 2: We explicitly `del agent_state_ref` before calling `ray.put` so
    # Ray can reclaim the previous state-dict from the object store promptly
    # instead of keeping two copies alive at once.
    #
    # FIX 3: `_ray_get_in_batches` collects results in small chunks so that all
    # 320 obs dicts (≈2 GB for --grid-repeats 16 × 20 cells) never land in
    # driver memory simultaneously.
    # -----------------------------------------------------------------
    agent_state_ref = None  # initialised on first loop iteration

    for update_idx in range(num_updates):
        p_hi = args.p_hi
        t0 = time.perf_counter()

        # ---- dispatch this update's rollouts (after weights are current) ----
        cells_cur = _build_cell_list()
        if agent_state_ref is not None:
            del agent_state_ref          # FIX 2: release previous Ray object ref
        agent_state_ref = ray.put({k: v.detach().cpu() for k, v in agent.state_dict().items()})
        pending_futures = _dispatch_rollouts(cells_cur, agent_state_ref, update_idx)

        # ---- collect in small batches to avoid ~2 GB simultaneous spike ----
        rollout_results = _ray_get_in_batches(pending_futures, batch_size=max(W * 4, 32))  # FIX 3

        # ---- unpack results ----
        (batch_xy_pos, batch_goal_vec, batch_chosen_bandits_motor,
         batch_bandit_obs, batch_chosen_bandits, batch_bandit_rewards,
         batch_meta_ep_start, batch_actor, batch_trial_cond,
         batch_teacher_correct, batch_logp_old, batch_v_old,
         cum_rewards_list_il, cum_rewards_list_ao, cum_rewards_list_ol,
         cum_high_reward_choices_il, cum_high_reward_choices_ao,
         cum_high_reward_choices_ol) = _collect_rollouts(rollout_results)
        
        t_rollout = time.perf_counter() - t0


        var_loss, motor_loss = _call_update2(
            agent, optim_bandit, optim_motor,
            batch_xy_pos,
            batch_goal_vec,
            batch_chosen_bandits_motor,
            batch_bandit_obs,
            batch_chosen_bandits,
            batch_bandit_rewards,
            batch_meta_ep_start,
            batch_actor,
            batch_trial_cond,
            batch_teacher_correct,
            batch_logp_old, 
            batch_v_old,
            device,
            ppo_epochs=args.ppo_epochs,
            ppo_minibatch_size=args.ppo_minibatch_size,
            aux_mb_size=args.aux_mb_size,
            self_bce_coef=args.self_bce_coef,
            aux_ao_coef=args.aux_ao_coef,
            aux_ol_coef=args.aux_ol_coef,
            gate_ao_coef=args.gate_ao_coef,
            gate_ol_coef=args.gate_ol_coef,
            gate_ol_sg=args.gate_ol_sg,
            session_chunk_size=args.train_session_chunk_size,
        )

        t_update = time.perf_counter() - (t0 + t_rollout)


        mean_cum_rew_il = float(np.mean(cum_rewards_list_il))
        mean_cum_rew_ao = float(np.mean(cum_rewards_list_ao))
        mean_cum_rew_ol = float(np.mean(cum_rewards_list_ol))

        # print("cum_high_reward_choices_il.shape = ", np.array(cum_high_reward_choices_il).shape)
        # print(cum_high_reward_choices_il)


        mean_hi_choices_il = np.mean([arr.sum() for arr in cum_high_reward_choices_il])
        mean_hi_choices_ao = np.mean([arr.sum() for arr in cum_high_reward_choices_ao])
        mean_hi_choices_ol = np.mean([arr.sum() for arr in cum_high_reward_choices_ol])

        #train_score = mean_hi_choices_il + mean_hi_choices_ao + mean_hi_choices_ol

        if update_idx % 10 == 0:
            print(f"[upd {update_idx:04d}] var_loss={var_loss:.4f} motor_loss={motor_loss:.4f} "
                  f"mean_hi_choices_il={mean_hi_choices_il:.1f} mean_hi_choices_ao={mean_hi_choices_ao:.1f} mean_hi_choices_ol={mean_hi_choices_ol:.1f} "
                  f"mean_cum_rew_il={mean_cum_rew_il:.1f} mean_cum_rew_ao={mean_cum_rew_ao:.1f} mean_cum_rew_ol={mean_cum_rew_ol:.1f}")
            print(f"[timing] rollout={t_rollout:.1f}s  update={t_update:.1f}s")


        # Grid-eval + early stopping (after warmup)
        if (update_idx >= args.warmup_updates) and (update_idx % args.eval_interval == 0):
            agent_state_cpu = {k: v.detach().cpu() for k, v in agent.state_dict().items()}

            p_lo_grid = P_LO_GRID_DEFAULT
            teacher_modes = TEACHER_MODES_DEFAULT

            macro, robust, eval_score, il_mean, ao_mean, ol_mean = eval_grid_score(
                workers=workers,
                agent_state_cpu=agent_state_cpu,
                session_K=session_K,
                p_hi=args.p_hi,
                p_lo_grid=p_lo_grid,
                teacher_modes=teacher_modes,
                n_sessions_per_cell=args.eval_sessions_per_cell,
                seed=args.seed + 10_000 + update_idx,
                alpha=0.5,
            )

            # Use a composite score to avoid "good on easy cells only"
            # eval_score = 0.7 * macro + 0.3 * worst5
            # eval_score = il_mean + ao_mean + ol_mean  # simpler unweighted sum of means

            print(f"[eval upd {update_idx:04d}] il_mean={il_mean:.2f} ao_mean={ao_mean:.2f} ol_mean={ol_mean:.2f} macro={macro:.2f} worst5={worst5:.2f} eval_score={eval_score:.2f}")

            writer.add_scalar("EvalGrid/macro", macro, update_idx)
            writer.add_scalar("EvalGrid/robust", robust, update_idx)
            writer.add_scalar("EvalGrid/score", eval_score, update_idx)

            if eval_score > best_eval + args.min_delta:
                best_eval = eval_score
                bad_evals = 0
                save_checkpoint(
                    os.path.join(run_dir, "best.pt"),
                    agent, optim_bandit, optim_motor,
                    extra={
                        "update": update_idx,
                        "eval_score": float(eval_score),
                        "macro": float(macro),
                        "robust": float(robust),
                        "p_lo_grid": list(p_lo_grid),
                        "teacher_modes": [m for m, _ in teacher_modes],
                    }
                )
            else:
                bad_evals += 1

            if bad_evals >= args.patience_evals:
                print(f"Early stopping at update {update_idx}, best_eval={best_eval:.2f}")
                break

        # TensorBoard
        writer.add_scalar("Loss/ChoiceLoss", float(var_loss), update_idx)
        writer.add_scalar("Loss/MotorLoss", float(motor_loss), update_idx)
        writer.add_scalar("Reward/MeanCumSession_IL", mean_cum_rew_il, update_idx)
        writer.add_scalar("Reward/MeanCumSession_AO", mean_cum_rew_ao, update_idx)
        writer.add_scalar("Reward/MeanCumSession_OL", mean_cum_rew_ol, update_idx)

    # -----------------------------
    # Load best checkpoint
    # -----------------------------

    ckpt_path = os.path.join(run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_extra = ckpt.get("extra", {})
    best_update = best_extra.get("update", -1)
    best_score = best_extra.get("eval_score", best_extra.get("ema_score", float("nan")))
    print(f"Loaded best checkpoint from update {best_update}, best_score={best_score:.2f}")
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    agent_state_cpu = {k: v.detach().cpu() for k, v in agent.state_dict().items()}

    # # -----------------------------
    # # Fixed-prob eval + per-trial plot
    # # -----------------------------
    # eval_il = []
    # eval_ao = []
    # eval_ol = []

    # mean_cum_rew_il = 0.0
    # mean_cum_rew_ao = 0.0
    # mean_cum_rew_ol = 0.0

    # rng_eval = np.random.default_rng(args.seed + 999)
    # agent_state_cpu = {k: v.detach().cpu() for k, v in agent.state_dict().items()}

    # probs_eval = [(0.2, 0.8) if rng_eval.random() < 0.5 else (0.8, 0.2) for _ in range(session_K)]

    # futures = []
    # for i in range(args.eval_sessions):
    #     futures.append(
    #         workers[i % len(workers)].run_session.remote(
    #             agent_state_cpu,
    #             probs_this_session=probs_eval,
    #             print_this_session=False,
    #             return_obs=False
    #         )
    #     )
    # results = ray.get(futures)

    # for res in results:
    #     (
    #         _, _, _,
    #         _, _, _, _,
    #         _, _,
    #         _,  # teacher_correct_buf
    #         cum_rew_il, cum_rew_ao, cum_rew_ol,
    #         pct_hi_il, pct_hi_ao, pct_hi_ol,
    #         high_reward_choices_il,
    #         high_reward_choices_ao,
    #         high_reward_choices_ol,
    #         _, _,  # logp_old_buf, v_old_buf
    #         _, _   # g_trace_ao, g_trace_ol
    #     ) = res

    #     eval_il.append(high_reward_choices_il)
    #     eval_ao.append(high_reward_choices_ao)
    #     eval_ol.append(high_reward_choices_ol)

    #     mean_cum_rew_il += cum_rew_il
    #     mean_cum_rew_ao += cum_rew_ao
    #     mean_cum_rew_ol += cum_rew_ol

    # mean_il = np.array(eval_il).mean(axis=0)
    # mean_ao = np.array(eval_ao).mean(axis=0)
    # mean_ol = np.array(eval_ol).mean(axis=0)

    # mean_cum_rew_il /= args.eval_sessions
    # mean_cum_rew_ao /= args.eval_sessions
    # mean_cum_rew_ol /= args.eval_sessions

    # print(f"Eval results over {args.eval_sessions} sessions:")
    # print(f" Mean cum reward IL: {mean_cum_rew_il:.1f}, AO: {mean_cum_rew_ao:.1f}, OL: {mean_cum_rew_ol:.1f}")
    # print(f" Mean high-reward choices per session IL: {mean_il.sum():.1f}, AO: {mean_ao.sum():.1f}, OL: {mean_ol.sum():.1f}")

    # plot_dir = os.path.join(args.save_dir, "plots")
    # os.makedirs(plot_dir, exist_ok=True)

    # run_dir = os.path.join(plot_dir, datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S"))
    # os.makedirs(run_dir, exist_ok=True)


    # fig, ax = plt.subplots(figsize=(6, 4))
    # trials = np.arange(session_N)
    # ax.plot(trials, mean_il, label="IL")
    # ax.plot(trials, mean_ao, label="AO")
    # ax.plot(trials, mean_ol, label="OL")
    # ax.set_xlabel("Trial index")
    # ax.set_ylabel("Mean high-reward choice")
    # ax.set_title("Per-trial performance (best checkpoint)")
    # ax.legend()
    # ax.grid(True)

    # plot_path = os.path.join(run_dir, "PerTrialReward.png")
    # fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    # writer.add_figure("Eval/PerTrialReward", fig)
    # plt.close(fig)



    # -----------------------------
    # Post-training grid eval (optional): teacher modes × reward probs
    # -----------------------------
    if args.post_grid_eval:
        p_hi = float(args.p_hi)
        # p_lo_grid = np.linspace(float(args.p_lo_max), float(args.p_lo_min), int(args.p_lo_steps))
        p_lo_grid = P_LO_GRID_DEFAULT  # fixed p_lo values for clearer plots (can be overridden by args)
        teacher_modes = TEACHER_MODES_DEFAULT

        print(f"[PostGrid] Evaluating {len(p_lo_grid)} p_lo values × {len(teacher_modes)} teacher modes = {len(p_lo_grid)*len(teacher_modes)} cells")
        print(f"[PostGrid] Sessions per cell: {int(args.post_grid_sessions_per_cell)}")

        cell_stats = eval_post_grid(
            workers=workers,
            agent_state_cpu=agent_state_cpu,
            session_K=session_K,
            p_hi=p_hi,
            p_lo_grid=[float(x) for x in p_lo_grid],
            teacher_modes=teacher_modes,
            n_sessions_per_cell=int(args.post_grid_sessions_per_cell),
            seed=int(args.seed + 40_000 + best_update),
        )

        # ------------------
        # Save CSV (one row per cell)
        # ------------------
        csv_path = os.path.join(run_dir, "PostGridEval.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(
                "p_hi,p_lo,gap,teacher_mode,"
                "pct_hi_il_mean,pct_hi_il_std,pct_hi_ao_mean,pct_hi_ao_std,pct_hi_ol_mean,pct_hi_ol_std,"
                "cum_rew_il_mean,cum_rew_il_std,cum_rew_ao_mean,cum_rew_ao_std,cum_rew_ol_mean,cum_rew_ol_std,"
                "gtrace_ao_mean_over_trials,gtrace_ol_mean_over_trials\n"
            )
            for p_lo in p_lo_grid:
                for mode_name, _cfg in teacher_modes:
                    k = (float(p_lo), str(mode_name))
                    if k not in cell_stats:
                        continue
                    st = cell_stats[k]
                    f.write(
                        f"{p_hi:.4f},{float(p_lo):.4f},{(p_hi-float(p_lo)):.4f},{mode_name},"
                        f"{st['pct_hi_il_mean']:.4f},{st['pct_hi_il_std']:.4f},{st['pct_hi_ao_mean']:.4f},{st['pct_hi_ao_std']:.4f},{st['pct_hi_ol_mean']:.4f},{st['pct_hi_ol_std']:.4f},"
                        f"{st['cum_rew_il_mean']:.4f},{st['cum_rew_il_std']:.4f},{st['cum_rew_ao_mean']:.4f},{st['cum_rew_ao_std']:.4f},{st['cum_rew_ol_mean']:.4f},{st['cum_rew_ol_std']:.4f},"
                        f"{st['gtrace_ao_mean_over_trials']:.4f},{st['gtrace_ol_mean_over_trials']:.4f}\n"
                    )

        # ------------------
        # Save raw g traces (mean/std/se) per cell
        # ------------------
        npz_path = os.path.join(run_dir, "PostGrid_gTrace_arrays.npz")
        npz_payload: Dict[str, Any] = {}
        for p_lo in p_lo_grid:
            for mode_name, _cfg in teacher_modes:
                k = (float(p_lo), str(mode_name))
                st = cell_stats.get(k)
                if st is None:
                    continue
                tag = f"mode={mode_name}__pLo={float(p_lo):.2f}"
                npz_payload[f"gAO_mean__{tag}"] = np.asarray(st["gtrace_ao_mean"], dtype=np.float32)
                npz_payload[f"gAO_std__{tag}"]  = np.asarray(st["gtrace_ao_std"],  dtype=np.float32)
                npz_payload[f"gAO_se__{tag}"]   = np.asarray(st["gtrace_ao_se"],   dtype=np.float32)
                npz_payload[f"gOL_mean__{tag}"] = np.asarray(st["gtrace_ol_mean"], dtype=np.float32)
                npz_payload[f"gOL_std__{tag}"]  = np.asarray(st["gtrace_ol_std"],  dtype=np.float32)
                npz_payload[f"gOL_se__{tag}"]   = np.asarray(st["gtrace_ol_se"],   dtype=np.float32)
                npz_payload[f"n_sessions__{tag}"] = np.array(st["n_sessions"], dtype=np.int32)
        if len(npz_payload) > 0:
            np.savez(npz_path, **npz_payload)

        # ------------------
        # Save comprehensive per-trial / performance arrays for notebook analysis
        # Includes: pertrial hit rates, cumulative pick rate, regret, TTC,
        #           and gate calibration (reliability targets) per cell.
        # Key format: {metric}__{tag}   tag = "mode={mode}__pLo={p_lo:.2f}"
        # ------------------
        perf_npz_path = os.path.join(run_dir, "PostGrid_perf_arrays.npz")
        perf_payload: Dict[str, Any] = {}

        _TTC_THRESHOLDS = [0.70, 0.80, 0.90]

        def _compute_ttc(sm: np.ndarray, thresholds) -> np.ndarray:
            """First trial index where smoothed mean >= each threshold; session_N if never."""
            out = []
            for thr in thresholds:
                idx = np.where(sm >= thr)[0]
                out.append(int(idx[0]) if len(idx) > 0 else int(session_N))
            return np.array(out, dtype=np.int32)

        def _rolling(x: np.ndarray, w: int = 5) -> np.ndarray:
            if len(x) < w or w <= 1:
                return x.copy()
            kernel = np.ones(w) / w
            return np.convolve(np.pad(x, w // 2, mode="edge"), kernel, mode="valid")[:len(x)]

        for p_lo in p_lo_grid:
            for mode_name, _cfg in teacher_modes:
                k = (float(p_lo), str(mode_name))
                st = cell_stats.get(k)
                if st is None:
                    continue
                tag = f"mode={mode_name}__pLo={float(p_lo):.2f}"
                gap = p_hi - float(p_lo)

                # ── per-trial hit rates (mean and SE, shape (T,)) ──
                for cond in ("il", "ao", "ol"):
                    perf_payload[f"pertrial_{cond}_mean__{tag}"] = np.asarray(st[f"pertrial_{cond}_mean"], dtype=np.float32)
                    perf_payload[f"pertrial_{cond}_se__{tag}"]   = np.asarray(st[f"pertrial_{cond}_se"],   dtype=np.float32)

                # ── cumulative pick rate (T,): running mean of per-trial hit rate ──
                for cond in ("il", "ao", "ol"):
                    raw  = np.nan_to_num(np.asarray(st[f"pertrial_{cond}_mean"], np.float32), nan=0.5)
                    denom = np.arange(1, len(raw) + 1, dtype=np.float32)
                    cum = np.cumsum(raw) / denom
                    perf_payload[f"cum_pickrate_{cond}__{tag}"] = cum.astype(np.float32)

                # ── regret = 1 − P(high-reward choice), smoothed (T,) ──
                for cond in ("il", "ao", "ol"):
                    raw = np.nan_to_num(np.asarray(st[f"pertrial_{cond}_mean"], np.float32), nan=0.5)
                    perf_payload[f"regret_{cond}__{tag}"] = (1.0 - _rolling(raw)).astype(np.float32)

                # ── TTC: first trial >= threshold, shape (3,) one value per threshold ──
                for cond in ("il", "ao", "ol"):
                    raw = np.nan_to_num(np.asarray(st[f"pertrial_{cond}_mean"], np.float32), nan=0.5)
                    sm  = _rolling(raw)
                    perf_payload[f"ttc_{cond}__{tag}"] = _compute_ttc(sm, _TTC_THRESHOLDS)

                # ── TTC threshold values (scalar array, same for all cells) ──
                perf_payload["ttc_thresholds"] = np.array(_TTC_THRESHOLDS, dtype=np.float32)

                # ── scalar summary stats (wrapped as 0-d or 1-d arrays) ──
                for stat in ("pct_hi_il_mean", "pct_hi_il_std",
                             "pct_hi_ao_mean", "pct_hi_ao_std",
                             "pct_hi_ol_mean", "pct_hi_ol_std",
                             "cum_rew_il_mean", "cum_rew_ao_mean", "cum_rew_ol_mean"):
                    perf_payload[f"{stat}__{tag}"] = np.array(st[stat], dtype=np.float32)

                perf_payload[f"n_sessions__{tag}"] = np.array(st["n_sessions"], dtype=np.int32)
                perf_payload[f"gap__{tag}"]        = np.array(gap, dtype=np.float32)

                # ── gate calibration: ao_rel_target / ol_rel_target if present ──
                if "ao_rel_target_mean" in st:
                    perf_payload[f"ao_rel_target_mean__{tag}"] = np.asarray(st["ao_rel_target_mean"], dtype=np.float32)
                    perf_payload[f"ol_rel_target_mean__{tag}"] = np.asarray(st["ol_rel_target_mean"], dtype=np.float32)
                # always save gate means for calibration subplot
                perf_payload[f"gAO_mean__{tag}"] = np.asarray(st["gtrace_ao_mean"], dtype=np.float32)
                perf_payload[f"gAO_se__{tag}"]   = np.asarray(st["gtrace_ao_se"],   dtype=np.float32)
                perf_payload[f"gOL_mean__{tag}"] = np.asarray(st["gtrace_ol_mean"], dtype=np.float32)
                perf_payload[f"gOL_se__{tag}"]   = np.asarray(st["gtrace_ol_se"],   dtype=np.float32)

        if perf_payload:
            np.savez(perf_npz_path, **perf_payload)
            print(f"[PostGrid] Saved performance arrays → {perf_npz_path}")

        # ------------------
        # Helper: rolling mean smoothing (preserves endpoints)
        # ------------------
        def _smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
            if len(x) < w:
                return x.copy()
            kernel = np.ones(w) / w
            pad = w // 2
            return np.convolve(
                np.pad(x, pad, mode="edge"), kernel, mode="valid"
            )[:len(x)]

        trials = np.arange(session_N)

        for p_lo in p_lo_grid:
            for mode_name, _cfg in teacher_modes:
                k = (float(p_lo), str(mode_name))
                if k not in cell_stats:
                    continue
                st = cell_stats[k]
                gap = p_hi - float(p_lo)

                # ----------------------------------------------------------
                # 1. Per-trial curve  (smoothed mean ± SE, raw mean faint behind)
                # ----------------------------------------------------------
                il_raw = np.nan_to_num(np.asarray(st["pertrial_il_mean"], dtype=np.float32), nan=0.5)
                ao_raw = np.nan_to_num(np.asarray(st["pertrial_ao_mean"], dtype=np.float32), nan=0.5)
                ol_raw = np.nan_to_num(np.asarray(st["pertrial_ol_mean"], dtype=np.float32), nan=0.5)
                il_se  = np.asarray(st["pertrial_il_se"], dtype=np.float32)
                ao_se  = np.asarray(st["pertrial_ao_se"], dtype=np.float32)
                ol_se  = np.asarray(st["pertrial_ol_se"], dtype=np.float32)
                W_sm = 5
                il_sm = _smooth(il_raw, W_sm)
                ao_sm = _smooth(ao_raw, W_sm)
                ol_sm = _smooth(ol_raw, W_sm)
                # SE also smoothed so bands match the smoothed mean
                il_se_sm = _smooth(il_se, W_sm)
                ao_se_sm = _smooth(ao_se, W_sm)
                ol_se_sm = _smooth(ol_se, W_sm)

                fig, ax = plt.subplots(figsize=(7, 4))
                # raw (faint background)
                ax.plot(trials, il_raw, color="C0", alpha=0.20, lw=1)
                ax.plot(trials, ao_raw, color="C1", alpha=0.20, lw=1)
                ax.plot(trials, ol_raw, color="C2", alpha=0.20, lw=1)
                # ±1 SE band on smoothed values
                ax.fill_between(trials,
                    np.clip(il_sm - il_se_sm, 0, 1), np.clip(il_sm + il_se_sm, 0, 1),
                    color="C0", alpha=0.15)
                ax.fill_between(trials,
                    np.clip(ao_sm - ao_se_sm, 0, 1), np.clip(ao_sm + ao_se_sm, 0, 1),
                    color="C1", alpha=0.15)
                ax.fill_between(trials,
                    np.clip(ol_sm - ol_se_sm, 0, 1), np.clip(ol_sm + ol_se_sm, 0, 1),
                    color="C2", alpha=0.15)
                # smoothed mean (bold)
                ax.plot(trials, il_sm, color="C0", lw=2, label="IL")
                ax.plot(trials, ao_sm, color="C1", lw=2, label="AO")
                ax.plot(trials, ol_sm, color="C2", lw=2, label="OL")
                ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.4, label="chance")
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("Trial index (within pair)")
                ax.set_ylabel("P(high-reward choice)")
                ax.set_title(f"Per-trial (mode={mode_name}, p_lo={float(p_lo):.2f}, gap={gap:.2f})")
                ax.grid(True, alpha=0.4)
                ax.legend(fontsize=9)
                fig.savefig(
                    os.path.join(run_dir, f"PostGrid_PerTrial_mode-{mode_name}_pLo-{float(p_lo):.2f}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

                # ----------------------------------------------------------
                # 2. Cumulative pick-rate curve  (monotonically informative)
                # ----------------------------------------------------------
                il_cum = np.cumsum(il_raw) / (trials + 1)
                ao_cum = np.cumsum(ao_raw) / (trials + 1)
                ol_cum = np.cumsum(ol_raw) / (trials + 1)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(trials, il_cum, label="IL",  color="C0", lw=2)
                ax.plot(trials, ao_cum, label="AO",  color="C1", lw=2)
                ax.plot(trials, ol_cum, label="OL",  color="C2", lw=2)
                ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.4, label="chance")
                ax.set_ylim(0.4, 1.02)
                ax.set_xlabel("Trial index (within pair)")
                ax.set_ylabel("Cumulative P(high-reward choice)")
                ax.set_title(f"Cumulative pick-rate (mode={mode_name}, p_lo={float(p_lo):.2f}, gap={gap:.2f})")
                ax.grid(True, alpha=0.4)
                ax.legend(fontsize=9)
                fig.savefig(
                    os.path.join(run_dir, f"PostGrid_CumPickRate_mode-{mode_name}_pLo-{float(p_lo):.2f}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

                # ----------------------------------------------------------
                # 3. Regret curve  (normalised: 0 = optimal, 1 = always-wrong)
                # regret_t = (p_hi - mean_reward_t) / (p_hi - p_lo)
                # We proxy mean_reward_t ≈ p_hi * P(high) + p_lo * (1 - P(high))
                # so regret = p_hi - [p_hi*P + p_lo*(1-P)] / gap
                #           = (p_hi - p_lo)*(1 - P) / (p_hi - p_lo) = 1 - P(high)
                # → normalised regret_t = 1 - P(high-reward choice at t)
                # ----------------------------------------------------------
                if gap > 0:
                    reg_il = 1.0 - _smooth(il_raw, W_sm)
                    reg_ao = 1.0 - _smooth(ao_raw, W_sm)
                    reg_ol = 1.0 - _smooth(ol_raw, W_sm)

                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.plot(trials, reg_il, color="C0", lw=2, label="IL")
                    ax.plot(trials, reg_ao, color="C1", lw=2, label="AO")
                    ax.plot(trials, reg_ol, color="C2", lw=2, label="OL")
                    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.4, label="chance")
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_xlabel("Trial index (within pair)")
                    ax.set_ylabel("Normalised regret  (1 − P(high))")
                    ax.set_title(f"Regret (mode={mode_name}, p_lo={float(p_lo):.2f}, gap={gap:.2f})")
                    ax.grid(True, alpha=0.4)
                    ax.legend(fontsize=9)
                    fig.savefig(
                        os.path.join(run_dir, f"PostGrid_Regret_mode-{mode_name}_pLo-{float(p_lo):.2f}.png"),
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(fig)

                # ----------------------------------------------------------
                # 4. Time-to-criterion: first trial where smoothed P ≥ threshold
                #    for three thresholds. Plotted as a simple table/bar chart.
                # ----------------------------------------------------------
                thresholds = [0.70, 0.80, 0.90]
                ttc = {"IL": [], "AO": [], "OL": []}
                for thr in thresholds:
                    for lbl, sm in [("IL", il_sm), ("AO", ao_sm), ("OL", ol_sm)]:
                        idx_arr = np.where(sm >= thr)[0]
                        ttc[lbl].append(int(idx_arr[0]) if len(idx_arr) > 0 else session_N)

                fig, ax = plt.subplots(figsize=(6, 4))
                x = np.arange(len(thresholds))
                w = 0.25
                for i, (lbl, color) in enumerate([("IL", "C0"), ("AO", "C1"), ("OL", "C2")]):
                    ax.bar(x + i * w, ttc[lbl], width=w, label=lbl, color=color)
                ax.set_xticks(x + w)
                ax.set_xticklabels([f"≥{int(t*100)}%" for t in thresholds])
                ax.set_ylabel("First trial index")
                ax.set_ylim(0, session_N + 1)
                ax.set_title(f"Time-to-criterion (mode={mode_name}, p_lo={float(p_lo):.2f})")
                ax.axhline(session_N, color="k", ls="--", lw=0.8, alpha=0.4, label=f"never ({session_N})")
                ax.grid(True, axis="y", alpha=0.4)
                ax.legend(fontsize=9)
                fig.tight_layout()
                fig.savefig(
                    os.path.join(run_dir, f"PostGrid_TTC_mode-{mode_name}_pLo-{float(p_lo):.2f}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

                # ----------------------------------------------------------
                # 5. Pick-rate bar (existing, kept)
                # ----------------------------------------------------------
                fig, ax = plt.subplots(figsize=(6, 4))
                means = [st["pct_hi_il_mean"], st["pct_hi_ao_mean"], st["pct_hi_ol_mean"]]
                stds  = [st["pct_hi_il_std"],  st["pct_hi_ao_std"],  st["pct_hi_ol_std"]]
                x = np.arange(3)
                ax.bar(x, means, yerr=stds, capsize=4, color=["C0", "C1", "C2"])
                ax.set_xticks(x)
                ax.set_xticklabels(["IL", "AO", "OL"])
                ax.set_ylabel("High-arm pick rate (%)")
                ax.set_ylim(0, 110)
                ax.set_title(f"High-arm pick rate (mode={mode_name}, p_lo={float(p_lo):.2f})")
                ax.grid(True, axis="y", alpha=0.4)
                fig.savefig(
                    os.path.join(run_dir, f"PostGrid_PctHi_mode-{mode_name}_pLo-{float(p_lo):.2f}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

                # ----------------------------------------------------------
                # 6. Gate g over trials  (±1 SE — uncertainty on the mean, not session variability)
                # ----------------------------------------------------------
                gao_m  = np.asarray(st["gtrace_ao_mean"], dtype=np.float32)
                gol_m  = np.asarray(st["gtrace_ol_mean"], dtype=np.float32)
                gao_se = np.asarray(st["gtrace_ao_se"],   dtype=np.float32)
                gol_se = np.asarray(st["gtrace_ol_se"],   dtype=np.float32)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(trials, gao_m, color="C0", lw=2, label="g (AO)")
                ax.fill_between(
                    trials,
                    np.clip(gao_m - gao_se, 0.0, 1.0),
                    np.clip(gao_m + gao_se, 0.0, 1.0),
                    color="C0", alpha=0.25,
                )
                ax.plot(trials, gol_m, color="C1", lw=2, label="g (OL)")
                ax.fill_between(
                    trials,
                    np.clip(gol_m - gol_se, 0.0, 1.0),
                    np.clip(gol_m + gol_se, 0.0, 1.0),
                    color="C1", alpha=0.25,
                )
                ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.3)
                ax.set_ylim(-0.02, 1.02)
                ax.set_xlabel("Trial index (within pair)")
                ax.set_ylabel("Gate g  (±1 SE across sessions)")
                ax.set_title(f"Gate g over trials (mode={mode_name}, p_lo={float(p_lo):.2f}, gap={gap:.2f})")
                ax.grid(True, alpha=0.4)
                ax.legend(fontsize=9)
                fig.savefig(
                    os.path.join(run_dir, f"PostGrid_gTrace_mode-{mode_name}_pLo-{float(p_lo):.2f}.png"),
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)

                # ----------------------------------------------------------
                # 7. Gate g vs teacher reliability calibration
                #    X-axis: trial index; two axes: g (left) and rel_target (right)
                #    rel_target computed from cell_stats if stored, else skip.
                # ----------------------------------------------------------
                if "ao_rel_target_mean" in st:
                    fig, ax1 = plt.subplots(figsize=(7, 4))
                    ax2 = ax1.twinx()
                    ax1.plot(trials, gao_m,  color="C0", lw=2, label="g_ao")
                    ax1.plot(trials, gol_m,  color="C1", lw=2, label="g_ol")
                    ax2.plot(trials, np.asarray(st["ao_rel_target_mean"]), color="C0",
                             lw=1.5, ls="--", label="rel_ao (target)")
                    ax2.plot(trials, np.asarray(st["ol_rel_target_mean"]), color="C1",
                             lw=1.5, ls="--", label="rel_ol (target)")
                    ax1.set_ylim(-0.02, 1.02)
                    ax2.set_ylim(-0.02, 1.02)
                    ax1.set_xlabel("Trial index")
                    ax1.set_ylabel("Gate g")
                    ax2.set_ylabel("Running teacher reliability")
                    ax1.set_title(f"Gate vs reliability (mode={mode_name}, p_lo={float(p_lo):.2f})")
                    lines1, labs1 = ax1.get_legend_handles_labels()
                    lines2, labs2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="lower right")
                    ax1.grid(True, alpha=0.4)
                    fig.savefig(
                        os.path.join(run_dir, f"PostGrid_GateCalib_mode-{mode_name}_pLo-{float(p_lo):.2f}.png"),
                        dpi=150, bbox_inches="tight",
                    )
                    plt.close(fig)

        # ------------------
        # Summary plots: per-teacher curves + heatmaps over the grid
        # ------------------
        mode_names = [m for m, _ in teacher_modes]
        p_lo_list = [float(x) for x in p_lo_grid]
        gaps = np.array([p_hi - x for x in p_lo_list], dtype=np.float32)

        # per-mode curves
        for mode_name, _cfg in teacher_modes:
            il = []
            ao = []
            ol = []
            for p_lo in p_lo_list:
                st = cell_stats.get((p_lo, str(mode_name)))
                if st is None:
                    il.append(np.nan)
                    ao.append(np.nan)
                    ol.append(np.nan)
                else:
                    il.append(st["pct_hi_il_mean"])
                    ao.append(st["pct_hi_ao_mean"])
                    ol.append(st["pct_hi_ol_mean"])

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(gaps, il, label="IL")
            ax.plot(gaps, ao, label="AO")
            ax.plot(gaps, ol, label="OL")
            ax.set_xlabel("Gap (p_hi - p_lo)")
            ax.set_ylabel("High-arm pick rate (%)")
            ax.set_title(f"High-arm pick rate vs gap (mode={mode_name})")
            ax.grid(True)
            ax.legend()
            fig.savefig(os.path.join(run_dir, f"PostGrid_ModeCurve_{mode_name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # heatmaps: rows=p_lo, cols=mode
        def _heatmap(mat: np.ndarray, title: str, fname: str):
            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(mat, aspect="auto", origin="lower")
            ax.set_xticks(np.arange(len(mode_names)))
            ax.set_xticklabels(mode_names, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(p_lo_list)))
            ax.set_yticklabels([f"{x:.2f}" for x in p_lo_list])
            ax.set_xlabel("Teacher mode")
            ax.set_ylabel("p_lo")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.9)
            fig.savefig(os.path.join(run_dir, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)

        mat_il = np.full((len(p_lo_list), len(mode_names)), np.nan, dtype=np.float32)
        mat_ao = np.full_like(mat_il, np.nan)
        mat_ol = np.full_like(mat_il, np.nan)
        mat_g_ao = np.full_like(mat_il, np.nan)
        mat_g_ol = np.full_like(mat_il, np.nan)
        for i, p_lo in enumerate(p_lo_list):
            for j, mode_name in enumerate(mode_names):
                st = cell_stats.get((p_lo, str(mode_name)))
                if st is None:
                    continue
                mat_il[i, j] = st["pct_hi_il_mean"]
                mat_ao[i, j] = st["pct_hi_ao_mean"]
                mat_ol[i, j] = st["pct_hi_ol_mean"]
                mat_g_ao[i, j] = float(st.get("gtrace_ao_mean_over_trials", np.nan))
                mat_g_ol[i, j] = float(st.get("gtrace_ol_mean_over_trials", np.nan))

        _heatmap(mat_il, "High-arm pick rate (%) — IL", "PostGrid_Heatmap_PctHi_IL.png")
        _heatmap(mat_ao, "High-arm pick rate (%) — AO", "PostGrid_Heatmap_PctHi_AO.png")
        _heatmap(mat_ol, "High-arm pick rate (%) — OL", "PostGrid_Heatmap_PctHi_OL.png")
        _heatmap(mat_g_ao, "Gate g (mean over trials) — AO-s", "PostGrid_Heatmap_g_AO.png")
        _heatmap(mat_g_ol, "Gate g (mean over trials) — OL-s", "PostGrid_Heatmap_g_OL.png")

        # ------------------
        # Mega-summary: per teacher mode (per-trial mean±std across p_lo + pick-rate vs gap mean±std)
        # ------------------
        for mode_name, _cfg in teacher_modes:
            # gather per-cell arrays (skip missing)
            pertrial_il = []
            pertrial_ao = []
            pertrial_ol = []
            gtrace_ao = []
            gtrace_ol = []
            pct_il = []
            pct_ao = []
            pct_ol = []
            pct_il_std = []
            pct_ao_std = []
            pct_ol_std = []
            gaps_ok = []

            for p_lo in p_lo_list:
                st = cell_stats.get((p_lo, str(mode_name)))
                if st is None:
                    continue
                pertrial_il.append(np.asarray(st["pertrial_il_mean"], dtype=np.float32))
                pertrial_ao.append(np.asarray(st["pertrial_ao_mean"], dtype=np.float32))
                pertrial_ol.append(np.asarray(st["pertrial_ol_mean"], dtype=np.float32))
                gtrace_ao.append(np.asarray(st["gtrace_ao_mean"], dtype=np.float32))
                gtrace_ol.append(np.asarray(st["gtrace_ol_mean"], dtype=np.float32))
                pct_il.append(float(st["pct_hi_il_mean"]))
                pct_ao.append(float(st["pct_hi_ao_mean"]))
                pct_ol.append(float(st["pct_hi_ol_mean"]))
                pct_il_std.append(float(st["pct_hi_il_std"]))
                pct_ao_std.append(float(st["pct_hi_ao_std"]))
                pct_ol_std.append(float(st["pct_hi_ol_std"]))
                gaps_ok.append(float(p_hi - p_lo))

            if len(pertrial_il) == 0:
                continue

            pertrial_il = np.stack(pertrial_il, axis=0)  # (P, N)
            pertrial_ao = np.stack(pertrial_ao, axis=0)
            pertrial_ol = np.stack(pertrial_ol, axis=0)

            gtrace_ao = np.stack(gtrace_ao, axis=0)
            gtrace_ol = np.stack(gtrace_ol, axis=0)

            # mean±std across p_lo (shows how trial curve changes with gap)
            il_mu = np.nanmean(pertrial_il, axis=0)
            ao_mu = np.nanmean(pertrial_ao, axis=0)
            ol_mu = np.nanmean(pertrial_ol, axis=0)
            il_sd = np.nanstd(pertrial_il, axis=0)
            ao_sd = np.nanstd(pertrial_ao, axis=0)
            ol_sd = np.nanstd(pertrial_ol, axis=0)

            gao_mu = np.nanmean(gtrace_ao, axis=0)
            gao_sd = np.nanstd(gtrace_ao, axis=0)
            gol_mu = np.nanmean(gtrace_ol, axis=0)
            gol_sd = np.nanstd(gtrace_ol, axis=0)

            # sort by gap for curve plot
            order = np.argsort(np.asarray(gaps_ok))
            gaps_s = np.asarray(gaps_ok)[order]
            pct_il_s = np.asarray(pct_il)[order]
            pct_ao_s = np.asarray(pct_ao)[order]
            pct_ol_s = np.asarray(pct_ol)[order]
            pct_il_sd_s = np.asarray(pct_il_std)[order]
            pct_ao_sd_s = np.asarray(pct_ao_std)[order]
            pct_ol_sd_s = np.asarray(pct_ol_std)[order]

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

            # left: per-trial mean±std across p_lo
            ax1.plot(trials, il_mu, label="IL")
            ax1.fill_between(trials, il_mu - il_sd, il_mu + il_sd, alpha=0.2)
            ax1.plot(trials, ao_mu, label="AO")
            ax1.fill_between(trials, ao_mu - ao_sd, ao_mu + ao_sd, alpha=0.2)
            ax1.plot(trials, ol_mu, label="OL")
            ax1.fill_between(trials, ol_mu - ol_sd, ol_mu + ol_sd, alpha=0.2)
            ax1.set_xlabel("Trial index")
            ax1.set_ylabel("Mean high-reward choice")
            ax1.set_title(f"Per-trial mean±std across p_lo (mode={mode_name})")
            ax1.grid(True)
            ax1.legend()

            # right: pick-rate vs gap mean±std across sessions (std per cell)
            ax2.plot(gaps_s, pct_il_s, label="IL")
            ax2.fill_between(gaps_s, pct_il_s - pct_il_sd_s, pct_il_s + pct_il_sd_s, alpha=0.2)
            ax2.plot(gaps_s, pct_ao_s, label="AO")
            ax2.fill_between(gaps_s, pct_ao_s - pct_ao_sd_s, pct_ao_s + pct_ao_sd_s, alpha=0.2)
            ax2.plot(gaps_s, pct_ol_s, label="OL")
            ax2.fill_between(gaps_s, pct_ol_s - pct_ol_sd_s, pct_ol_s + pct_ol_sd_s, alpha=0.2)
            ax2.set_xlabel("Gap (p_hi - p_lo)")
            ax2.set_ylabel("High-arm pick rate (%)")
            ax2.set_title(f"Pick rate vs gap mean±std (mode={mode_name})")
            ax2.grid(True)
            ax2.legend()

            # third: gate g over trials (mean±std across p_lo)
            ax3.plot(trials, gao_mu, label="g (AO)")
            ax3.fill_between(trials, gao_mu - gao_sd, gao_mu + gao_sd, alpha=0.2)
            ax3.plot(trials, gol_mu, label="g (OL)")
            ax3.fill_between(trials, gol_mu - gol_sd, gol_mu + gol_sd, alpha=0.2)
            ax3.set_ylim(-0.05, 1.05)
            ax3.set_xlabel("Trial index")
            ax3.set_ylabel("Gate g")
            ax3.set_title(f"Gate g mean±std across p_lo (mode={mode_name})")
            ax3.grid(True)
            ax3.legend()

            fig.suptitle(f"Post-grid mega-summary — {mode_name} (p_hi={p_hi:.2f})", y=1.02)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"PostGrid_MegaSummary_Mode_{mode_name}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)


    # Cleanup
    ray.shutdown()
    writer.close()


if __name__ == "__main__":
    main()