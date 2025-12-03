import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import visual_bandit_env2 as vbe
import var_bandit_learner2 as bl
from bandit_train_batch import extract_pair_view

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- build env & agent (same hyperparams as training) ---
env = vbe.TwoChoiceReachingEnv(
    W=384,
    H=400,
    render_mode="rgb_array",
    seed=0,
    session_K=3,
    session_N=12,
    randomize_sides=False,
)

feature_dim = 64
input_size = feature_dim + 2 + 1 + 1
hidden = 128
action_dim = 2

agent = bl.BanditLearner(
    input_size=input_size,
    feature_dim=feature_dim,
    rnn_hidden_size=hidden,
    action_dim=action_dim,
).to(device)
agent.eval()    # we only use encoder

# optionally: load trained checkpoint
# load_checkpoint("your_ckpt.pt", agent, map_location=device)

X = []
y = []

num_episodes = 100   # enough to see the pattern

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False

    while not done:
        # same preprocessing as in rollout
        obs_tensor = torch.from_numpy(obs).to(device).permute(2, 0, 1).unsqueeze(0).float()
        obs_tensor.mul_(1.0 / 255.0)

        pair_view = extract_pair_view(obs_tensor, env, crop_size=112, pad=6)  # (1,3,112,224)
        feat = agent.encode(pair_view).squeeze(0).detach().cpu().numpy()      # (F,)

        pair_idx = info.get("pair_index_in_session", -1)

        X.append(feat)
        y.append(pair_idx)

        # take any action; here just stand still
        action = np.zeros(2, np.float32)
        obs, reward, term, trunc, info = env.step(action)
        done = bool(term) or bool(trunc)

X = torch.tensor(np.stack(X), dtype=torch.float32)
y = torch.tensor(np.array(y), dtype=torch.long)

num_pairs = int(y.max().item()) + 1
print("Feature dataset:", X.shape, "num_pairs:", num_pairs)


perm = torch.randperm(len(X))
train_n = int(0.8 * len(X))
train_idx, test_idx = perm[:train_n], perm[train_n:]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

clf = nn.Linear(feature_dim, num_pairs).to(device)
opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

for epoch in range(200):
    opt.zero_grad()
    logits = clf(X_train.to(device))
    loss = F.cross_entropy(logits, y_train.to(device))
    loss.backward()
    opt.step()

with torch.no_grad():
    train_acc = (clf(X_train.to(device)).argmax(dim=1) == y_train.to(device)).float().mean().item()
    test_acc  = (clf(X_test.to(device)).argmax(dim=1)  == y_test.to(device)).float().mean().item()

print(f"train acc={train_acc:.3f}, test acc={test_acc:.3f}")
