import ray
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
from bandit_train_batch import RolloutWorker as rw
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
import math
import pygame
import os
from datetime import datetime

from gymnasium.wrappers import RecordVideo
from torch.utils.tensorboard import SummaryWriter


import math, contextlib
import matplotlib.pyplot as plt


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
warmup_updates = 500


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
ray.init(ignore_reinit_error=True)

# ---------- rollout workers ----------
workers = [
    rw.remote(session_K, session_N, seed=42, worker_id=i)
    for i in range(8)
]


ckpt = torch.load(os.path.join("checkpoints", "best.pt"), map_location=device)
print(f"Loaded best checkpoint from update {ckpt['extra']['update']}, ema_score={ckpt['extra']['ema_score']:.2f}")
agent.load_state_dict(ckpt["model_state"])
agent.eval()

eval_il = []
eval_ao = []
eval_ol = []

mean_cum_rew_il = 0.0
mean_cum_rew_ao = 0.0
mean_cum_rew_ol = 0.0

num_eval_sessions = 500 # larger number for stable eval

for i in range(num_eval_sessions):
    res = ray.get(
        workers[0].run_session.remote(
            {k: v.cpu() for k, v in agent.state_dict().items()},
            probs_this_session=[(0.8, 0.2)] * session_K,
            print_this_session=True
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

plot_dir = os.path.join("checkpoints", "plots")
os.makedirs(plot_dir, exist_ok=True)

# save to file
plot_path = os.path.join(plot_dir, "PerTrialReward_pretrained2.png")
fig.savefig(plot_path, dpi=150, bbox_inches="tight")


writer.add_figure("Eval/PerTrialReward", fig)
plt.close(fig)


ray.shutdown()
writer.close()