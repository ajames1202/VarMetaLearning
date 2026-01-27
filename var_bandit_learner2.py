import numpy as np
import torch
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F
import math
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class CNNEncoder(nn.Module):
    """A small CNN for processing visual inputs (expects NCHW)."""
    def __init__(self, feature_dim):
        super().__init__()
        # build the CNN feature extractor
        self.cnn = self.__build_cnn(3, feature_dim)

    def __build_cnn(self, in_channels, feature_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, feature_dim),
        )

    def forward(self, x):
        # x: (N,C,H,W)
        return self.cnn(x)


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))


def log_prob_tanh_normal(action, mean, log_std, eps=1e-6):
    """Log prob under tanh-squashed Normal with reparam correction."""
    std = torch.exp(log_std)
    # undo tanh
    pretahn_actions = torch.atanh(action.clamp(-1 + eps, 1 - eps))
    # Normal log prob
    normal = torch.distributions.Normal(mean, std)
    log_prob = normal.log_prob(pretahn_actions).sum(dim=-1)
    # correction
    corr = torch.log(1 - torch.tanh(pretahn_actions).pow(2) + 1e-6).sum(dim=-1)
    return log_prob - corr

def lr_repulsion_loss(left_feats, right_feats, margin=0.2):
    # left_feats/right_feats: (T,B,F)
    lf = left_feats.reshape(-1, left_feats.size(-1))
    rf = right_feats.reshape(-1, right_feats.size(-1))
    lf = F.normalize(lf, dim=-1)
    rf = F.normalize(rf, dim=-1)
    sim = (lf * rf).sum(dim=-1)          # cosine similarity in [-1,1]
    return F.relu(sim - margin).mean()   # penalize if too similar


def report_param_set(params, label):
    params = list(params)
    touched = 0
    total = 0
    for p in params:
        total += 1
        if p.grad is not None and float(p.grad.abs().sum()) != 0.0:
            touched += 1
    print(f"{label}: touched {touched}/{total} params")

def debug_left_right_feats(left_feats, right_feats, tag=""):
    # left_feats/right_feats: (T,B,F)
    with torch.no_grad():
        lf = left_feats.reshape(-1, left_feats.size(-1))   # (N,F)
        rf = right_feats.reshape(-1, right_feats.size(-1)) # (N,F)

        # normalize for cosine
        lf_n = F.normalize(lf, dim=-1)
        rf_n = F.normalize(rf, dim=-1)

        cos_pair = (lf_n * rf_n).sum(dim=-1)            # (N,)
        l2_pair  = (lf - rf).norm(dim=-1)               # (N,)

        # random pairing baseline
        perm = torch.randperm(rf.size(0), device=rf.device)
        cos_rand = (lf_n * rf_n[perm]).sum(dim=-1)
        l2_rand  = (lf - rf[perm]).norm(dim=-1)

        print(f"[{tag}] cosine paired   mean/std: {cos_pair.mean():.3f} / {cos_pair.std():.3f}")
        print(f"[{tag}] cosine random   mean/std: {cos_rand.mean():.3f} / {cos_rand.std():.3f}")
        print(f"[{tag}] L2 paired       mean/std: {l2_pair.mean():.3f} / {l2_pair.std():.3f}")
        print(f"[{tag}] L2 random       mean/std: {l2_rand.mean():.3f} / {l2_rand.std():.3f}")


import numpy as np
import torch

def get_time_batch_indices_for_label(labels_tb: np.ndarray, target: str, device):
    """
    labels_tb: (T,B) numpy array of strings
    Returns:
      t_grid: (K,B) LongTensor (time indices per batch, sorted)
      b_grid: (K,B) LongTensor (batch indices)
      K: int
      B: int
    Assumes each batch has the same number K of occurrences of target.
    """
    assert labels_tb.ndim == 2, f"Expected (T,B), got {labels_tb.shape}"
    T, B = labels_tb.shape

    # times_per_b[b] = array of time indices where labels_tb[:,b] == target
    times_per_b = [np.where(labels_tb[:, b] == target)[0] for b in range(B)]
    if max((len(x) for x in times_per_b), default=0) == 0:
        raise ValueError(f"No occurrences of target='{target}' found.")

    # Enforce constant K across batches (required to form a (K,B) grid)
    K = len(times_per_b[0])
    for b in range(B):
        if len(times_per_b[b]) != K:
            raise ValueError(
                f"Unequal counts for '{target}': batch0={K}, batch{b}={len(times_per_b[b])}. "
                f"Either balance counts or handle ragged K."
            )

    t_grid_np = np.stack([np.sort(times_per_b[b]) for b in range(B)], axis=1)  # (K,B)
    b_grid_np = np.broadcast_to(np.arange(B)[None, :], (K, B))                 # (K,B)

    t_grid = torch.as_tensor(t_grid_np, device=device, dtype=torch.long)
    b_grid = torch.as_tensor(b_grid_np, device=device, dtype=torch.long)
    return t_grid, b_grid, K, B



def subset_TBH_grouped(tensor_tbh, t_grid, b_grid):
    return tensor_tbh[t_grid, b_grid]  # (K,B,...)



class BanditLearner(nn.Module):
    def __init__(self, input_size, feature_dim, rnn_hidden_size, action_dim,
                 log_std_min=-5.0, log_std_max=-1.0):
        nn.Module.__init__(self)

        self.enc = CNNEncoder(feature_dim)
        self.rnn_hidden_size = rnn_hidden_size

        # --- Bandit RNN + heads
        
        self.enc_self = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, dropout=0.1)
        self.enc_teacher = nn.MultiheadAttention(embed_dim=rnn_hidden_size, num_heads=4, dropout=0.1)

        self.z_dim = 16   # start here (or 8/32)

        self.prior_head = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, 2*self.z_dim)
        )

        self.post_head  = nn.Sequential(nn.Linear(rnn_hidden_size, rnn_hidden_size),
                                         nn.ReLU(), 
                                         nn.Linear(rnn_hidden_size, 2*self.z_dim))


        self.dec_teacher = nn.MultiheadAttention(embed_dim=self.z_dim, num_heads=4, dropout=0.1)

        self.action_dec = nn.MultiheadAttention(embed_dim=self.z_dim, num_heads=4, dropout=0.1)
        self.reward_dec = nn.MultiheadAttention(embed_dim=self.z_dim, num_heads=4, dropout=0.1)

        self.H_to_z = nn.Linear(rnn_hidden_size, self.z_dim)

        self.a_logits = nn.Linear(self.z_dim, 2)
        self.r_logits = nn.Linear(self.z_dim, 1)


        self.attn_pos_emb = nn.Embedding(12, rnn_hidden_size)  # max 12 trials per episode
        
        # self.q_in = nn.Linear(2*feature_dim, rnn_hidden_size, bias=False)
        self.inp_emb = nn.Linear(2*feature_dim, rnn_hidden_size, bias=False)

        self.arm_reward_head = nn.Sequential(
            nn.Linear(rnn_hidden_size + feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        
        self.actor = nn.Embedding(2, rnn_hidden_size)
        self.action = nn.Embedding(2, rnn_hidden_size)
        self.rwd_in = nn.Embedding(3, rnn_hidden_size)  # 0,1 , No_reward

        self.attn_ln = nn.LayerNorm(rnn_hidden_size)
        self.attn_ln2 = nn.LayerNorm(self.z_dim)
        self.teacher_ln = nn.LayerNorm(rnn_hidden_size)


        # pol_net_inp = 4 * rnn_hidden_size + rnn_hidden_size #(x_t.size = 4*H, z_t.size = H)
        self.policy_net_il  = nn.Sequential(nn.Linear(self.rnn_hidden_size + self.z_dim, 128), nn.ReLU(), nn.Linear(128, 2))
        self.policy_net_obs = nn.Sequential(nn.Linear(self.rnn_hidden_size + 2*self.z_dim, 128), nn.ReLU(), nn.Linear(128, 2))

        self.critic_net_il  = nn.Sequential(nn.Linear(self.z_dim, 128), nn.ReLU(), nn.Linear(128, 1))


        self.critic_net_obs = nn.Sequential(
            nn.Linear(2*self.z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # motor policy
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

    def bandit_parameters(self):
        # everything that should be updated by bandit_loss
        modules = []
        modules += [
            self.attn_ln,
            self.attn_pos_emb,
            # self.ctx_to_logit,
            self.arm_reward_head,
            self.enc,
            self.actor,
            self.action,
            self.rwd_in,
            self.enc_self, 
            self.enc_teacher,
            self.post_head, 
            self.prior_head,
            self.dec_teacher, 
            self.action_dec, 
            self.reward_dec,
            self.H_to_z, 
            self.a_logits, 
            self.r_logits,
            self.teacher_ln, 
            self.attn_ln2,
            self.policy_net_il,
            self.policy_net_obs,
            self.critic_net_il,
            self.critic_net_obs
            # self.fam_head,   # if you use familiarity as bandit aux
        ]
        for m in modules:
            if m is None:
                continue
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
            if m is None:
                continue
            for p in m.parameters():
                yield p

    def encode(self, x):
        # x = self.downsample(x)  # (T, C, H, W)
        return self.enc(x)

    
    def reward_compute(self, h, left_feats, right_feats):
        # h: (T,B,H) or (1,H)
        l = sigmoid(self.arm_reward_head(torch.cat([h, left_feats],  dim=-1)))
        r = sigmoid(self.arm_reward_head(torch.cat([h, right_feats], dim=-1)))

        return l, r  # (...,2)

    def motor_fwd(self, choice_target, xy_pos=None, goal_vec=None):
        pos_emb  = self.mlp_pos(xy_pos)
        goal_emb = self.mlp_goal(goal_vec)
        motor_inp  = torch.cat([choice_target, pos_emb, goal_emb], dim=-1)
        mu      = self.mu_head(motor_inp)
        log_std = self.log_std_head(motor_inp).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

   
    def update2(self, optim_bandit, optim_motor, xy_pos_buf, goal_vec_buf, chosen_bandits_motor_buf,
        bandit_obs,           # list of dicts with "left"/"right"
        chosen_bandits_buf,   # list of (T, 2) tensors
        bandit_rewards_buf,   # list of (T,) tensors
        meta_ep_start_buf,    # list of (T,) tensors
        actor_buf,
        trial_cond_buf,
        device,
    ):
        # -----------------------------
        # 1) MOTOR data: keep flat (no batch)
        # -----------------------------
        xy_pos   = torch.as_tensor(np.stack(xy_pos_buf), device=device, dtype=torch.float32)
        goalvec  = torch.as_tensor(np.asarray(goal_vec_buf, np.float32), device=device)
        chosen_bandits_motor = torch.as_tensor(
            np.stack(chosen_bandits_motor_buf), device=device, dtype=torch.float32
        )

        # -----------------------------
        # 2) BANDIT data: batched (B, T, ...)
        # -----------------------------
        left_obs  = torch.stack([torch.tensor(b["left"],  device=device, dtype=torch.float32) for b in bandit_obs], dim=0)  # (B,T,C,H,W)
        right_obs = torch.stack([torch.tensor(b["right"], device=device, dtype=torch.float32) for b in bandit_obs], dim=0)  # (B,T,C,H,W)
        chosen_bandits  = torch.stack(chosen_bandits_buf, dim=0).to(device)     # (B,T,2)
        rewards_bandits = torch.stack(bandit_rewards_buf, dim=0).to(device)     # (B,T)
        meta_ep_start   = torch.stack(meta_ep_start_buf, dim=0).to(device)      # (B,T)
        actor = torch.stack(actor_buf, dim=0).to(device)                      # (B,T)
        trial_cond = np.array(trial_cond_buf)      # (B,T)

        B, T, C, H, W = left_obs.shape

        # time-major
        chosen_bandits  = chosen_bandits.permute(1, 0, 2)   # (T,B,2)
        rewards_bandits = rewards_bandits.permute(1, 0)     # (T,B)
        meta_ep_start   = meta_ep_start.permute(1, 0)       # (T,B)

        rewards_rnn = rewards_bandits.unsqueeze(-1)  # (T,B,1)
        start_rnn   = meta_ep_start.unsqueeze(-1)    # (T,B,1)

        actor = actor.permute(1,0)  # (T,B)
        trial_cond = trial_cond.transpose(1,0)  # (T,B)

        num_epochs = 1
        var_loss_sum, var_slices = 0.0, 0
        motor_loss_sum, motor_slices = 0.0, 0
        # left_feats, right_feats = None 
        
        T = actor.shape[0]
        device = actor.device

        t_il_idx, b_il_idx, K, B = get_time_batch_indices_for_label(trial_cond, "IL", device=device)
        t_ao_o_idx, b_ao_o_idx, K, B = get_time_batch_indices_for_label(trial_cond, "AO-o", device=device)
        t_ao_s_idx, b_ao_s_idx, K, B = get_time_batch_indices_for_label(trial_cond, "AO-s", device=device)
        t_ol_o_idx, b_ol_o_idx, K, B = get_time_batch_indices_for_label(trial_cond, "OL-o", device=device)
        t_ol_s_idx, b_ol_s_idx, K, B = get_time_batch_indices_for_label(trial_cond, "OL-s", device=device)

        z_final  = torch.zeros(T, B, self.z_dim, device=device)
        x_final  = torch.zeros(T, B, self.rnn_hidden_size, device=device)  # before H_to_z
        act_final= torch.zeros(T, B, device=device, dtype=torch.long)
        rew_final= torch.zeros(T, B, device=device)  # or (T,B,1)

        for _ in range(num_epochs):
            optim_motor.zero_grad(set_to_none=True)
            optim_bandit.zero_grad(set_to_none=True)

            # -----------------------------
            # 3) BANDIT loss (query/outcome interleaving)
            # -----------------------------

            left_flat  = left_obs.reshape(B*T, C, H, W)
            right_flat = right_obs.reshape(B*T, C, H, W)

            left_feats  = self.encode(left_flat).view(B, T, -1).permute(1, 0, 2)   # (T,B,F)
            right_feats = self.encode(right_flat).view(B, T, -1).permute(1, 0, 2)  # (T,B,F)

            
            aL = chosen_bandits[..., 0:1]
            aR = chosen_bandits[..., 1:2]

            chosen_feats   = aL * left_feats  + aR * right_feats
            # unchosen_feat = aL * right_feats + aR * left_feats

            ## ACTION-ONLY Trials: use action, actor, reward embeddings as history vals
            left_feats_ao_o = subset_TBH_grouped(left_feats,  t_ao_o_idx, b_ao_o_idx)
            right_feats_ao_o = subset_TBH_grouped(right_feats,  t_ao_o_idx, b_ao_o_idx)
            l_r_concat_ao = torch.cat([left_feats_ao_o, right_feats_ao_o], dim=-1)
            inp_emb_ao = self.inp_emb(l_r_concat_ao)  # (T,B,H)


            chosen_bandits_ao_o = subset_TBH_grouped(chosen_bandits,  t_ao_o_idx, b_ao_o_idx)
            actor_ao_o = subset_TBH_grouped(actor,  t_ao_o_idx, b_ao_o_idx)
            rewards_ao_o = subset_TBH_grouped(rewards_rnn,  t_ao_o_idx, b_ao_o_idx)
            action_ao_o = torch.argmax(chosen_bandits_ao_o, dim=-1)  # (T,B)
            action_emb_ao = self.action(action_ao_o)
            actor_emb_ao = self.actor(actor_ao_o)
            rwd_emb_ao = self.rwd_in(rewards_ao_o.squeeze(-1).long())
            # val_ao = action_emb_ao + actor_emb_ao + rwd_emb_ao  # (T,B,H)
            x_ao_o = inp_emb_ao + action_emb_ao + actor_emb_ao + rwd_emb_ao  # (T,B,2F)
            x_ao_o_dec = inp_emb_ao + actor_emb_ao
           
            L = x_ao_o.size(0)  # query length (T-1)
            # mask future keys: shape (L, L), True means "masked", all keys > t are masked
            attn_mask_obs = torch.triu(torch.ones((L, L), device=x_ao_o.device, dtype=torch.bool), diagonal=1)

            z_ao_out, w = self.enc_teacher(x_ao_o, x_ao_o, x_ao_o, attn_mask=attn_mask_obs)
            z_ao_out = self.teacher_ln(z_ao_out + x_ao_o)   

            q_params_ao = self.post_head(z_ao_out)                       # (T,B,2*z_dim)
            mu_q_ao, log_std_q_ao = q_params_ao[..., :self.z_dim], q_params_ao[..., self.z_dim:]
            log_std_q_ao = log_std_q_ao.clamp(-5.0, 2.0)

            eps = torch.randn_like(mu_q_ao)
            z_ao_o = mu_q_ao + log_std_q_ao.exp() * eps                   # (T,B,z_dim)
            z_final[t_ao_o_idx, b_ao_o_idx] = z_ao_o
            x_final[t_ao_o_idx, b_ao_o_idx] = x_ao_o_dec
            act_final[t_ao_o_idx, b_ao_o_idx] = action_ao_o
            rew_final[t_ao_o_idx, b_ao_o_idx] = rewards_ao_o.squeeze(-1)

            # Predict u_ao(t+1) from ua_ao(t)
            attn_mask_Lm1 = torch.triu(torch.ones((L-1, L-1), device=x_ao_o.device, dtype=torch.bool), diagonal=1)
            u_ao_tplus1,_ = self.dec_teacher(self.H_to_z(x_ao_o[:-1]), z_ao_o[:-1], z_ao_o[:-1], attn_mask=attn_mask_Lm1) # (T-1,B,H)
            u_ao_tplus1 = self.attn_ln2(u_ao_tplus1)

            p_params = self.prior_head(u_ao_tplus1)                      # (T-1,B,2*z_dim)
            mu_prior_ao, log_std_prior_ao = p_params[..., :self.z_dim], p_params[..., self.z_dim:]
            log_std_prior_ao = log_std_prior_ao.clamp(-5.0, 2.0)

            #Recontrunction loss for AO
            # attn_mask_obs_tmin1 = torch.triu(torch.ones((L, L), device=z_ao_o.device, dtype=torch.bool), diagonal=0)
            x_ao_o_z = self.H_to_z(x_ao_o)
            L, B, Dz = x_ao_o_z.shape

            bos = torch.zeros(1, B, Dz, device=x_ao_o_z.device)      # or learnable parameter
            kv  = torch.cat([bos, x_ao_o_z[:-1]], dim=0)             # shift right by 1

            # standard causal mask (allows self-attend)
            attn_mask = torch.triu(
                torch.ones((L, L), device=x_ao_o_z.device, dtype=torch.bool),
                diagonal=1
            )

            out_ao_act, _ = self.action_dec(z_ao_o, kv, kv, attn_mask=attn_mask)
            logits_a_ao = self.a_logits(out_ao_act)

            ao_choice_loss = F.cross_entropy(logits_a_ao.reshape(-1,2), action_ao_o.reshape(-1))
            

            # out_ao_r,_ = self.reward_dec(z_ao_o, x_ao_o_z, x_ao_o_z, attn_mask=attn_mask_obs_tmin1)  ## Use an attention module ?
            # logits_ao_r = self.r_logits(out_ao_r)
            # ao_reward_loss = F.binary_cross_entropy_with_logits(logits_ao_r, rewards_ao_o.float())

            #KL loss
            log_std_q_ao = log_std_q_ao[1:]
            mu_q_ao = mu_q_ao[1:]
            kl_err_ao = log_std_q_ao - log_std_prior_ao + (torch.pow(torch.exp(log_std_q_ao),2) + torch.pow(mu_q_ao-mu_prior_ao,2))/(2*torch.pow(torch.exp(log_std_prior_ao),2)) - 0.5
            kl_err_ao = kl_err_ao.sum(-1).mean()


            ## OL Trials: use only history (no current obs)
            left_feats_ol_o = subset_TBH_grouped(left_feats,  t_ol_o_idx, b_ol_o_idx)
            right_feats_ol_o = subset_TBH_grouped(right_feats,  t_ol_o_idx, b_ol_o_idx)
            chosen_bandits_ol_o = subset_TBH_grouped(chosen_bandits,  t_ol_o_idx, b_ol_o_idx)
            actor_ol_o = subset_TBH_grouped(actor,  t_ol_o_idx, b_ol_o_idx)
            rewards_ol_o = subset_TBH_grouped(rewards_rnn,  t_ol_o_idx, b_ol_o_idx)

            l_r_concat_ol = torch.cat([left_feats_ol_o, right_feats_ol_o], dim=-1)
            inp_emb_ol = self.inp_emb(l_r_concat_ol)  # (T,B,H)

            action_ol_o = torch.argmax(chosen_bandits_ol_o, dim=-1)  # (T,B)
            action_emb_ol = self.action(action_ol_o)
            actor_emb_ol = self.actor(actor_ol_o)
            rwd_emb_ol = self.rwd_in(rewards_ol_o.squeeze(-1).long())
            # val_ol = action_emb_ol + actor_emb_ol + rwd_emb_ol  # (T,B,H)
            x_ol_o = inp_emb_ol + action_emb_ol + actor_emb_ol + rwd_emb_ol
            x_ol_o_dec = inp_emb_ol + actor_emb_ol

            L = x_ol_o.size(0)  # query length (T-1)
            # mask future keys: shape (L, L), True means "masked", all keys
            attn_mask_obs = torch.triu(torch.ones((L, L), device=x_ol_o.device, dtype=torch.bool), diagonal=1)
            z_ol_out, w = self.enc_teacher(x_ol_o, x_ol_o, x_ol_o, attn_mask=attn_mask_obs)
            z_ol_out = self.teacher_ln(z_ol_out + x_ol_o)

            q_params_ol = self.post_head(z_ol_out)                       # (T,B,2*z_dim)
            mu_q_ol, log_std_q_ol = q_params_ol[..., :self.z_dim], q_params_ol[..., self.z_dim:]
            log_std_q_ol = log_std_q_ol.clamp(-5.0, 2.0)
            eps = torch.randn_like(mu_q_ol)
            z_ol_o = mu_q_ol + log_std_q_ol.exp() * eps                   # (T,B,z_dim)
            z_final[t_ol_o_idx, b_ol_o_idx] = z_ol_o
            x_final[t_ol_o_idx, b_ol_o_idx] = x_ol_o_dec
            act_final[t_ol_o_idx, b_ol_o_idx] = action_ol_o
            rew_final[t_ol_o_idx, b_ol_o_idx] = rewards_ol_o.squeeze(-1)
            
            #Predict u_ol(t+1) from u_ol(t)
            attn_mask_Lm1 = torch.triu(torch.ones((L-1, L-1), device=x_ol_o.device, dtype=torch.bool), diagonal=1)
            u_ol_tplus1,_ = self.dec_teacher(self.H_to_z(x_ol_o[:-1]), z_ol_o[:-1], z_ol_o[:-1], attn_mask=attn_mask_Lm1) # (T-1,B,H)
            u_ol_tplus1 = self.attn_ln2(u_ol_tplus1)

            p_params_ol = self.prior_head(u_ol_tplus1)                      # (T-1,B,2*z_dim)
            mu_prior_ol, log_std_prior_ol = p_params_ol[..., :self.z_dim], p_params_ol[..., self.z_dim:]
            log_std_prior_ol = log_std_prior_ol.clamp(-5.0, 2.0)


            #Recontrunction loss for OL
            # attn_mask_obs_tmin1 = torch.triu(torch.ones((L, L), device=x_ol_o.device, dtype=torch.bool), diagonal=0)
            x_ol_o_z = self.H_to_z(x_ol_o)

            # out_ol_act,_ = self.action_dec(z_ol_o, x_ol_o_z, x_ol_o_z, attn_mask=attn_mask_obs_tmin1)  ## Use an attention module ?
            # logits_a_ol = self.a_logits(out_ol_act)

            bos = torch.zeros(1, B, Dz, device=x_ol_o_z.device)      # or learnable parameter
            kv  = torch.cat([bos, x_ol_o_z[:-1]], dim=0)             # shift right by 1

            # standard causal mask (allows self-attend)
            attn_mask = torch.triu(
                torch.ones((L, L), device=x_ol_o_z.device, dtype=torch.bool),
                diagonal=1
            )
            out_ol_act, _ = self.action_dec(z_ol_o, kv, kv, attn_mask=attn_mask)
            logits_a_ol = self.a_logits(out_ol_act)
            ol_choice_loss = F.cross_entropy(logits_a_ol.reshape(-1,2), action_ol_o.reshape(-1))
            

            # out_ol_r,_ = self.reward_dec(z_ol_o, x_ol_o_z, x_ol_o_z, attn_mask=attn_mask_obs_tmin1)  ## Use an attention module ?
            out_ol_r,_ = self.reward_dec(z_ol_o, kv, kv, attn_mask=attn_mask)  ## Use an attention module ?
            logits_r_ol = self.r_logits(out_ol_r)
            ol_reward_loss = F.binary_cross_entropy_with_logits(logits_r_ol, rewards_ol_o.float())

            #KL loss
            log_std_q_ol = log_std_q_ol[1:]
            mu_q_ol = mu_q_ol[1:]
            kl_err_ol = log_std_q_ol - log_std_prior_ol + (torch.pow(torch.exp(log_std_q_ol),2) + torch.pow(mu_q_ol-mu_prior_ol,2))/(2*torch.pow(torch.exp(log_std_prior_ol),2)) - 0.5
            kl_err_ol = kl_err_ol.sum(-1).mean()

            
            ### IL trials #############

            left_feats_il = subset_TBH_grouped(left_feats,  t_il_idx, b_il_idx)
            right_feats_il = subset_TBH_grouped(right_feats,  t_il_idx, b_il_idx)
            chosen_bandits_il = subset_TBH_grouped(chosen_bandits,  t_il_idx, b_il_idx)
            actor_il = subset_TBH_grouped(actor,  t_il_idx, b_il_idx)
            rewards_il = subset_TBH_grouped(rewards_rnn,  t_il_idx, b_il_idx)


            #self_trials = (trial_cond=="OL-s") | (trial_cond=="AO-s") | (trial_cond=="IL")
            l_r_concat_il = torch.concat([left_feats_il, right_feats_il], dim=-1)
            inp_emb_il = self.inp_emb(l_r_concat_il)  # (T,B,H)

            action_il = torch.argmax(chosen_bandits_il, dim=-1)  # (T,B)
            action_emb_il = self.action(action_il)
            actor_emb_il = self.actor(actor_il)
            rwd_emb_il = self.rwd_in(rewards_il.squeeze(-1).long())
            # val_ol = action_emb_ol + actor_emb_ol + rwd_emb_ol  # (T,B,H)
            x_il = inp_emb_il+ action_emb_il + actor_emb_il + rwd_emb_il
            x_il_dec = inp_emb_il + actor_emb_il

            L = x_il.size(0)  # query length (T-1)
            # mask future keys: shape (L, L), True means "masked", all keys
            attn_mask_obs = torch.triu(torch.ones((L, L), device=x_il.device, dtype=torch.bool), diagonal=1)
            z_il_out, w = self.enc_self(x_il, x_il, x_il, attn_mask=attn_mask_obs)

            q_params_il = self.post_head(z_il_out)                       # (T,B,2*z_dim)
            mu_q_il, log_std_q_il = q_params_il[..., :self.z_dim], q_params_il[..., self.z_dim:]
            log_std_q_il = log_std_q_il.clamp(-5.0, 2.0)
            eps = torch.randn_like(mu_q_il)
            z_il = mu_q_il + eps * torch.exp(log_std_q_il)
            z_final[t_il_idx, b_il_idx] = z_il
            x_final[t_il_idx, b_il_idx] = x_il_dec
            act_final[t_il_idx, b_il_idx] = action_il
            rew_final[t_il_idx, b_il_idx] = rewards_il.squeeze(-1)

                        
            ### For AO-s trials ####
            left_feats_ao_s = subset_TBH_grouped(left_feats,  t_ao_s_idx, b_ao_s_idx)
            right_feats_ao_s = subset_TBH_grouped(right_feats,  t_ao_s_idx, b_ao_s_idx)
            chosen_bandits_ao_s = subset_TBH_grouped(chosen_bandits,  t_ao_s_idx, b_ao_s_idx)
            actor_ao_s = subset_TBH_grouped(actor,  t_ao_s_idx, b_ao_s_idx)
            rewards_ao_s = subset_TBH_grouped(rewards_rnn,  t_ao_s_idx, b_ao_s_idx)

            l_r_concat_ao_s = torch.concat([left_feats_ao_s, right_feats_ao_s], dim=-1)
            inp_emb_ao_s = self.inp_emb(l_r_concat_ao_s)  # (T,B,H)

            action_ao_s = torch.argmax(chosen_bandits_ao_s, dim=-1)  # (T,B)
            action_emb_ao_s = self.action(action_ao_s)
            actor_emb_ao_s = self.actor(actor_ao_s)
            rwd_emb_ao_s = self.rwd_in(rewards_ao_s.squeeze(-1).long())
            # val_ol = action_emb_ol + actor_emb_ol + rwd_emb_ol  # (T,B,H)
            x_ao_s = inp_emb_ao_s + action_emb_ao_s + actor_emb_ao_s + rwd_emb_ao_s
            x_ao_s_dec = inp_emb_ao_s + actor_emb_ao_s
            L = x_ao_s.size(0)
            attn_mask_obs = torch.triu(torch.ones((L, L), device=x_ao_s.device, dtype=torch.bool), diagonal=1)
            z_ao_s_out, w = self.enc_self(x_ao_s, x_ao_s, x_ao_s, attn_mask=attn_mask_obs)

            q_params_ao_s = self.post_head(z_ao_s_out)                       # (T,B,2*z_dim)
            mu_q_ao_s, log_std_q_ao_s = q_params_ao_s[..., :self.z_dim], q_params_ao_s[..., self.z_dim:]
            log_std_q_ao_s = log_std_q_ao_s.clamp(-5.0, 2.0)
            eps = torch.randn_like(mu_q_ao_s)
            z_ao_s = mu_q_ao_s + eps * torch.exp(log_std_q_ao_s)
            z_final[t_ao_s_idx, b_ao_s_idx] = z_ao_s
            x_final[t_ao_s_idx, b_ao_s_idx] = x_ao_s_dec
            act_final[t_ao_s_idx, b_ao_s_idx] = action_ao_s
            rew_final[t_ao_s_idx, b_ao_s_idx] = rewards_ao_s.squeeze(-1)

            ### For OL trials ####
            left_feats_ol_s = subset_TBH_grouped(left_feats,  t_ol_s_idx, b_ol_s_idx)
            right_feats_ol_s = subset_TBH_grouped(right_feats,  t_ol_s_idx, b_ol_s_idx)
            chosen_bandits_ol_s = subset_TBH_grouped(chosen_bandits,  t_ol_s_idx, b_ol_s_idx)
            actor_ol_s = subset_TBH_grouped(actor,  t_ol_s_idx, b_ol_s_idx)
            rewards_ol_s = subset_TBH_grouped(rewards_rnn,  t_ol_s_idx, b_ol_s_idx)

            l_r_concat_ol_s = torch.concat([left_feats_ol_s, right_feats_ol_s], dim=-1)
            inp_emb_ol_s = self.inp_emb(l_r_concat_ol_s)  # (T,B,H)

            action_ol_s = torch.argmax(chosen_bandits_ol_s, dim=-1)  # (T,B)
            action_emb_ol_s = self.action(action_ol_s)
            actor_emb_ol_s = self.actor(actor_ol_s)
            rwd_emb_ol_s = self.rwd_in(rewards_ol_s.squeeze(-1).long())
            # val_ol = action_emb_ol + actor_emb_ol + rwd_emb_ol  # (T,B,H)
            x_ol_s = inp_emb_ol_s + action_emb_ol_s + actor_emb_ol_s + rwd_emb_ol_s
            x_ol_s_dec = inp_emb_ol_s + actor_emb_ol_s
            L = x_ol_s.size(0)
            attn_mask_obs = torch.triu(torch.ones((L, L), device=x_ol_s.device, dtype=torch.bool), diagonal=1)
            z_ol_s_out, w = self.enc_self(x_ol_s, x_ol_s, x_ol_s, attn_mask=attn_mask_obs)

            q_params_ol_s = self.post_head(z_ol_s_out)                       # (T,B,2*z_dim)
            mu_q_ol_s, log_std_q_ol_s = q_params_ol_s[..., :self.z_dim], q_params_ol_s[..., self.z_dim:]
            log_std_q_ol_s = log_std_q_ol_s.clamp(-5.0, 2.0)
            eps = torch.randn_like(mu_q_ol_s)
            z_ol_s = mu_q_ol_s + eps * torch.exp(log_std_q_ol_s)
            z_final[t_ol_s_idx, b_ol_s_idx] = z_ol_s
            x_final[t_ol_s_idx, b_ol_s_idx] = x_ol_s_dec
            act_final[t_ol_s_idx, b_ol_s_idx] = action_ol_s
            rew_final[t_ol_s_idx, b_ol_s_idx] = rewards_ol_s.squeeze(-1)


            ### Compute policy loss
            actor_loss = torch.zeros(T, B, device=device)
            critic_loss = torch.zeros(T, B, device=device)

            for b in range(B):
                z_ao_o_last_t = torch.zeros(self.z_dim, device=device)
                z_ol_o_last_t = torch.zeros(self.z_dim, device=device)
                z_il_tm1 = torch.zeros(self.z_dim, device=device)
                z_ao_s_tm1 = torch.zeros(self.z_dim, device=device)
                z_ol_s_tm1 = torch.zeros(self.z_dim, device=device)

                for t in range(T):
                    if trial_cond[t,b] == "IL":
                        policy_logits = self.policy_net_il(torch.concat([x_final[t,b], z_il_tm1], dim = -1)).squeeze(-1)
                        critic_val = self.critic_net_il(z_il_tm1).squeeze(-1)
                        z_il_tm1 = z_final[t,b]
                    elif trial_cond[t,b] == "AO-o":
                        z_ao_o_last_t = z_final[t,b]
                        continue
                    elif trial_cond[t,b] == "AO-s":
                        policy_logits = self.policy_net_obs(torch.concat([x_final[t,b], z_ao_s_tm1, z_ao_o_last_t], dim = -1)).squeeze(-1)
                        critic_val = self.critic_net_obs(torch.cat([z_ao_s_tm1, z_ao_o_last_t], dim=-1)).squeeze(-1)
                        z_ao_s_tm1 = z_final[t,b]
                    elif trial_cond[t,b] == "OL-o":
                        z_ol_o_last_t = z_final[t,b]
                        continue
                    elif trial_cond[t,b] == "OL-s":
                        policy_logits = self.policy_net_obs(torch.concat([x_final[t,b], z_ol_s_tm1, z_ol_o_last_t], dim = -1)).squeeze(-1)
                        critic_val = self.critic_net_obs(torch.cat([z_ol_s_tm1, z_ol_o_last_t], dim = -1)).squeeze(-1)
                        z_ol_s_tm1 = z_final[t,b]
                    
                    dist_t = torch.distributions.Categorical(logits=policy_logits)
                    logpi_t = dist_t.log_prob(act_final[t,b])
                    adv = (rew_final[t,b] - critic_val).detach()
                    actor_loss[t,b] += -(adv * logpi_t)
                    critic_loss[t,b] += (rew_final[t,b] - critic_val).pow(2)

      

            actor_loss = actor_loss.mean()
            critic_loss = critic_loss.mean()

            beh_loss = actor_loss + critic_loss + ao_choice_loss \
            + kl_err_ao + ol_choice_loss + ol_reward_loss + kl_err_ol    

            # (optional) KL term kept from your code (unused in final loss unless you re-enable it)
            eps = 1e-8
            # variational_loss.backward()
            contrast_loss = lr_repulsion_loss(left_feats, right_feats, margin=0.2)

            lambda_contrast = 0.05  # try 0.01–0.2
            bandit_total = beh_loss + lambda_contrast * contrast_loss
            bandit_total.backward()


            var_loss_sum += beh_loss.item()
            var_slices += 1

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



        # debug_left_right_feats(left_feats, right_feats, tag="enc")

        var_loss = var_loss_sum / max(1, var_slices)
        motor_loss = motor_loss_sum / max(1, motor_slices)
        return var_loss, motor_loss


