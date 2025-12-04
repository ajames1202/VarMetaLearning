import ray
import numpy as np
import torch
import visual_bandit_env2 as vbe
from bandit_train_batch import extract_pair_view
from var_bandit_learner2 import BanditLearner

import torch.nn.functional as F
from torch import nn

import var_bandit_learner2 as bl
from pathlib import Path
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@ray.remote
class FeatureWorker:
    def __init__(self, encoder_state_dict, env_kwargs, feature_dim, device="cpu", seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device(device)

        # build env
        self.env = vbe.TwoChoiceReachingEnv(**env_kwargs)

        # build agent but we only care about enc
        input_size = feature_dim + 2 + 1 + 1   # same as training
        hidden = 128
        action_dim = 2

        self.agent = BanditLearner(
            input_size=input_size,
            feature_dim=feature_dim,
            rnn_hidden_size=hidden,
            action_dim=action_dim,
        ).to(self.device)
        self.agent.load_state_dict(encoder_state_dict, strict=False)
        self.agent.eval()

    def collect(self, num_episodes, save_pair_views=False, save_dir="pair_views"):
        save_dir = Path(save_dir)
        if save_pair_views:
            save_dir.mkdir(parents=True, exist_ok=True)

        X_list = []
        y_list = []

        for _ in range(num_episodes):
            obs, info = self.env.reset()
            pair_idx = info.get("pair_index_in_session", -1)
            trial_idx = info.get("trial_index", -1)

            done = False
            ep_start_flag = 1.0

            while not done:
                obs_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float()
                obs_tensor = obs_tensor.to(self.device)

                pair_view = extract_pair_view(obs_tensor, self.env,
                                            crop_size=112, pad=6)
                pair_view.mul_(1.0 / 255.0)

                if save_pair_views and ep_start_flag == 1.0:
                    # pair_view: (1, C, H, W)
                    img = pair_view[0].detach().cpu()
                    img = (img * 255).clamp(0, 255).byte()
                    img = img.permute(1, 2, 0).numpy()

                    filename = save_dir / f"ep0_step{int(trial_idx):02d}_pair{int(pair_idx):02d}.png"
                    print(f"Trial={trial_idx}, Pair_idx={pair_idx}")
                    Image.fromarray(img).save(filename)

                feat = self.agent.encode(pair_view).squeeze(0).detach().cpu().numpy()

                pair_idx = info.get("pair_index_in_session", -1)

                X_list.append(feat)
                y_list.append(pair_idx)

                action = np.zeros(2, np.float32)
                obs, reward, term, trunc, info = self.env.step(action)
                trial_idx = info.get("trial_index", -1)

                if info.get("trial_ended", False):
                    print(f"Trial{trial_idx} ended.")
                    ep_start_flag = 1.0
                else:
                    ep_start_flag = 0.0    
                done = bool(term)

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)
        return X, y



import ray
import torch
import numpy as np
from var_bandit_learner2 import BanditLearner

ray.init()

# --- load trained agent once on the driver ---
feature_dim = 64
input_size = feature_dim + 2 + 1 + 1
hidden = 128
action_dim = 2

agent = BanditLearner(
    input_size=input_size,
    feature_dim=feature_dim,
    rnn_hidden_size=hidden,
    action_dim=action_dim,
)
agent.eval()

encoder_state = agent.state_dict()   # or agent.enc.state_dict() if you prefer

env_kwargs = dict(
    W=384,
    H=400,
    render_mode="rgb_array",
    seed=0,                # base seed, workers will offset this
    session_K=3,
    session_N=12,
    randomize_sides=False,
)

num_workers = 8
episodes_per_worker = 10

workers = [
    FeatureWorker.remote(
        encoder_state_dict=encoder_state,
        env_kwargs=env_kwargs,
        feature_dim=feature_dim,
        device="cpu",
        seed=1000 + i,
    )
    for i in range(num_workers)
]

# futures = [w.collect.remote(episodes_per_worker, save_pair_views =  ) for w in workers]
rollout_futures = []
for i, w in enumerate(workers):
    save_pair_views = (i == 0)
    rollout_futures.append(
        w.collect.remote(
            episodes_per_worker,
            save_pair_views=save_pair_views,
            save_dir="pair_views"
        )
    )



results = ray.get(rollout_futures)

X_parts = [r[0] for r in results]
y_parts = [r[1] for r in results]

X = np.concatenate(X_parts, axis=0)
y = np.concatenate(y_parts, axis=0)

num_pairs = int(y.max().item()) + 1
print("Feature dataset shape:", X.shape, "labels shape:", y.shape)


perm = torch.randperm(len(X))
train_n = int(0.8 * len(X))
train_idx, test_idx = perm[:train_n], perm[train_n:]

X_train, y_train = X[train_idx], y[train_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).long().to(device)
X_test  = torch.from_numpy(X_test).float().to(device)
y_test  = torch.from_numpy(y_test).long().to(device)


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


# mean feature per pair
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).long().to(device)

mu = []
for k in range(num_pairs):
    mu_k = X[y == k].mean(dim=0)
    mu.append(mu_k)
mu = torch.stack(mu)   # (num_pairs, F)

# within-pair variance
within = 0.0
count = 0
for k in range(num_pairs):
    diffs = X[y == k] - mu[k]
    within += (diffs.norm(dim=1) ** 2).sum()
    count  += diffs.size(0)
within = (within / count).sqrt().item()

# between-pair distances
M = mu.size(0)
between = []
for i in range(M):
    for j in range(i+1, M):
        between.append((mu[i] - mu[j]).norm().item())
between = float(np.mean(between))

print(f"within={within:.3f}, between={between:.3f}")

