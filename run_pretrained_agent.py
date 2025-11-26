import torch
import numpy as np

import visual_bandit_env2 as vbe          # your env
import var_bandit_learner2 as bl          # BanditLearner
from bandit_train_batch import load_checkpoint, meta_ep_rollout  # existing helpers


def main():
    # ---- 1. Config: MUST match training ----
    session_K = 3
    session_N = 20

    feature_dim = 128
    hidden_size = 128
    input_size = feature_dim + 2 + 1 + 1   # features + prev_action(2) + prev_reward(1) + meta_ep_start(1)

    # Path to the checkpoint you copied from the cluster, e.g.
    # scp user@cluster:/path/to/checkpoints/bandit_latest.pth .
    CHECKPOINT_PATH = "checkpoints/bandit_latest.pth"

    # ---- 2. Device (CPU or GPU if available) ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- 3. Recreate the agent with SAME hyperparams as training ----
    agent = bl.BanditLearner(
        input_size=input_size,
        feature_dim=feature_dim,
        rnn_hidden_size=hidden_size,
        action_dim=2,
        feature_source="ids",       # important: same as in training
        num_pairs=session_K,
        max_trials=session_N * session_K,
    ).to(device)

    # We don't need an optimizer for inference
    _extra = load_checkpoint(CHECKPOINT_PATH, agent, optimizer=None, map_location=device)
    agent.eval()
    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")

    # ---- 4. Create an environment for evaluation ----
    # Same settings as in RolloutWorker in bandit_train_batch.py
    env = vbe.TwoChoiceReachingEnv(
        W=384,
        H=400,
        render_mode="rgb_array",   # change to "human" if your env supports it and you want a window
        seed=0,
        session_K=session_K,
        session_N=session_N,
        trial_ms=3000,
        randomize_sides=False,
    )

    # Example bandit probabilities for this evaluation session
    # Here: either all pairs have (0.8, 0.2) or all (0.2, 0.8)
    if np.random.rand() < 0.5:
        L = [0.8] * session_K
    else:
        L = [0.2] * session_K
    R = [1.0 - x for x in L]
    probs_this_session = [(L[i], R[i]) for i in range(session_K)]
    env.unwrapped.pair_probs = probs_this_session

    print("Pair probabilities for this session:", probs_this_session)

    # ---- 5. Run ONE rollout with the loaded agent ----
    (
        xy_pos_buf,
        goal_vec_buf,
        feats_motor,
        chosen_bandits_motor_buf,
        feats_bandit,
        chosen_bandits_buf,
        bandit_rewards_buf,
        meta_ep_start_buf,
        high_reward_choice_per_N,
        cum_rewards,
    ) = meta_ep_rollout(
        env,
        agent,
        device,
        session_K=session_K,
        session_N=session_N,
        worker_id=0,
        print_this_session=True,   # will print detailed per-trial info
    )

    print("\n=== Evaluation summary ===")
    print("Total bandit rewards in session:", cum_rewards)
    print("High-reward choice fraction per N:", high_reward_choice_per_N)
    print("Number of trials:", len(bandit_rewards_buf))


if __name__ == "__main__":
    main()
