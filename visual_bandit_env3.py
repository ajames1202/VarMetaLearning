# visual_bandit_env.py
# A self-contained Gymnasium environment with "sessions":
# One Gym episode == one session. Each session generates K fresh image pairs.
# Each of those K pairs is randomly selected N times (shuffled schedule),
# then the episode ends. No images are reused across sessions.

from __future__ import annotations

import math
import numpy as np
import os
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["AUDIODEV"] = "null"

import pygame
from typing import Optional, List, Tuple

import gymnasium as gym
from gymnasium import spaces
import random



def _seeded_rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


class ProceduralImageFactory:
    """
    Very simple procedural "image" generator to produce pygame.Surface objects.
    You can swap this out for your own image pipeline. It generates abstract
    patterns so pairs are visually distinct between sessions.
    """

    def __init__(self, palette: Optional[List[Tuple[int, int, int]]] = None):
        self.palette = palette or [
            (240, 240, 240),
            (220, 220, 255),
            (255, 220, 220),
            (220, 255, 220),
            (255, 240, 200),
            (200, 240, 255),
        ]

    def generate(self, rng: np.random.Generator, size: Tuple[int, int]) -> pygame.Surface:
        w, h = size
        surf = pygame.Surface((w, h))
        # Background
        bg = tuple(int(c) for c in rng.integers(0, 200, size=3))
        surf.fill(bg)

        # Draw a few random shapes with random colors/positions
        n_shapes = rng.integers(4, 10)
        for _ in range(int(n_shapes)):
            color = self.palette[int(rng.integers(0, len(self.palette)))]
            shape_type = int(rng.integers(0, 3))  # 0=rect,1=circle,2=line
            if shape_type == 0:
                rw, rh = int(max(4, rng.normal(w * 0.2, w * 0.1))), int(max(4, rng.normal(h * 0.2, h * 0.1)))
                rx, ry = int(rng.integers(0, max(1, w - rw))), int(rng.integers(0, max(1, h - rh)))
                pygame.draw.rect(surf, color, pygame.Rect(rx, ry, rw, rh))
            elif shape_type == 1:
                r = int(max(2, rng.normal(min(w, h) * 0.12, min(w, h) * 0.05)))
                cx, cy = int(rng.integers(r, max(r + 1, w - r))), int(rng.integers(r, max(r + 1, h - r)))
                pygame.draw.circle(surf, color, (cx, cy), r)
            else:
                x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
                x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
                pygame.draw.line(surf, color, (x1, y1), (x2, y2), width=int(rng.integers(1, 4)))
        return surf
    
    # def generate(self, rng: np.random.Generator, size: Tuple[int, int]) -> pygame.Surface:
    #     w, h = size
    #     surf = pygame.Surface((w, h), flags=pygame.SRCALPHA)

    #     # Background
    #     bg = tuple(int(c) for c in rng.integers(0, 200, size=3))
    #     surf.fill(bg)

    #     # Build token 'A00'..'Z99'
    #     letter = chr(int(rng.integers(0, 26)) + ord('A'))
    #     number = int(rng.integers(0, 100))
    #     token = f"{letter}{number:02d}"

    #     # Pick a readable text color (use palette if present, otherwise auto-contrast)
    #     def lum(rgb):
    #         r, g, b = rgb
    #         return 0.2126 * r + 0.7152 * g + 0.0722 * b

    #     if getattr(self, "palette", None):
    #         color = self.palette[int(rng.integers(0, len(self.palette)))]
    #         # Ensure decent contrast with background
    #         if abs(lum(color) - lum(bg)) < 60:
    #             color = (255, 255, 255) if lum(bg) < 128 else (0, 0, 0)
    #     else:
    #         color = (255, 255, 255) if lum(bg) < 128 else (0, 0, 0)

    #     # Initialize font and size to fit nicely inside the surface
    #     if not pygame.font.get_init():
    #         pygame.font.init()
    #     max_fs = int(min(w, h) * 0.8)
    #     fs = max_fs
    #     font = pygame.font.Font(None, fs)
    #     text_surf = font.render(token, True, color)

    #     # Shrink text until it fits within 90% of both width and height
    #     while (text_surf.get_width() > 0.9 * w or text_surf.get_height() > 0.9 * h) and fs > 8:
    #         fs = max(8, int(fs * 0.9))
    #         font = pygame.font.Font(None, fs)
    #         text_surf = font.render(token, True, color)

    #     # Outline for readability
    #     outline_color = (0, 0, 0) if lum(color) > 128 else (255, 255, 255)
    #     outline_surf = font.render(token, True, outline_color)
    #     ow = max(1, int(fs * 0.08))  # outline width in pixels

    #     cx, cy = w // 2, h // 2
    #     for dx in (-ow, 0, ow):
    #         for dy in (-ow, 0, ow):
    #             if dx == 0 and dy == 0:
    #                 continue
    #             surf.blit(outline_surf, outline_surf.get_rect(center=(cx + dx, cy + dy)))

    #     # Blit the main text
    #     surf.blit(text_surf, text_surf.get_rect(center=(cx, cy)))

    #     return surf



class TwoChoiceReachingEnv(gym.Env):
    """
    TwoChoiceReaching-v0 (sessionized)
    ----------------------------------
    - One Gym episode == one *session*.
    - Each session generates K fresh image pairs (left/right panels at top).
    - Within the session, each of the K pairs is selected randomly N times.
    - Each selection defines a *trial*: the agent moves a cursor (vx,vy) each step.
      If the agent reaches the left or right target area within the time limit,
      it receives a Bernoulli reward drawn from that side's probability for the
      current pair. Then the next trial begins (until K*N trials are completed).
    - Images are not reused across sessions.

    Observation: RGB image (H, W, 3), dtype=uint8
    Action: Box(low=-1, high=1, shape=(2,), dtype=float32) for (vx, vy) per step
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        W: int = 512,
        H: int = 384,
        margin: int = 16,
        randomize_sides: bool = True,
        pair_probs: Optional[List[Tuple[float, float]]] = None,  # Optional fixed probs per pair
        # Target (panel) configuration
        target_w: int = 160,
        target_h: int = 120,
        target_gap: int = 24,
        # --- Session config ---
        session_K: int = 3,  # number of pairs per session
        session_N: int = 8,  # times each pair is selected per session (random order)
        # Back-compat aliases (override the NEW names if provided)
        n_pairs: Optional[int] = None,
        repeats_per_pair: Optional[int] = None,
        # Trial timing
        trial_ms: int = 2000,
        # Rendering
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        shuffle: bool = True,
        expert_teacher: bool = False,
        expert_eps: float = 0.1,
        unrealiable_teacher: bool = False,
        alpha: Optional[List] = None,
        tau: Optional[List] = None

    ):
        super().__init__()

        # Basic geometry
        self.W = int(W)
        self.H = int(H)
        self.margin = int(margin)
        self.randomize_sides = bool(randomize_sides)

        # Targets (image panels) geometry
        self.target_w = int(target_w)
        self.target_h = int(target_h)
        self.target_gap = int(target_gap)

        # Session params (support legacy names)
        if n_pairs is not None:
            session_K = n_pairs
        if repeats_per_pair is not None:
            session_N = repeats_per_pair
        self.session_K = int(session_K)
        self.session_N = int(session_N)

        # Probs and timing
        self.pair_probs = pair_probs  # if None, random per session (K pairs)
        self.trial_ms = int(trial_ms)
        self.steps_per_trial = max(1, int(np.ceil(self.trial_ms / 1000.0 * self.metadata["render_fps"])))

        # RNG
        self.rng = _seeded_rng(seed)

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.H, self.W, 3), dtype=np.uint8)

        # pygame setup (defer display creation)
        self._display = None
        self._window = None
        self.render_mode = render_mode

        # Runtime state
        self.image_factory = ProceduralImageFactory()
        self.session_pairs: List[Tuple[pygame.Surface, pygame.Surface]] = []  # (left,right) surfaces for this session
        self.session_id = 0

        self.shuffle = shuffle

        self.expert_teacher = expert_teacher
        self.eps = expert_eps  # for epsilon-greedy expert behavior in obs trials


        self.unrealiable_teacher = unrealiable_teacher

        assert not(unrealiable_teacher and expert_teacher), "unreliable_teacher and expert_teacher cannot both be True (contradictory modes)"

        self.alpha = alpha
        self.tau = tau 
        self._build_static_canvas()

    # ---------- helpers ----------

    def _build_static_canvas(self):
        pygame.init()
        self._backbuffer = pygame.Surface((self.W, self.H))
        self._font = pygame.font.SysFont(None, 18)

    def _ensure_display(self):
        if self.render_mode == "human" and self._window is None:
            # Create a window the first time we need to actually show something
            self._window = pygame.display.set_mode((self.W, self.H))
            pygame.display.set_caption("TwoChoiceReaching-v0 (sessionized)")

    def _compute_target_rects(self) -> Tuple[pygame.Rect, pygame.Rect]:
        # Place two panels at the top, centered horizontally with a gap
        total_w = self.target_w * 2 + self.target_gap
        left_x = (self.W - total_w) // 2
        right_x = left_x + self.target_w + self.target_gap
        y = self.margin
        left_rect = pygame.Rect(left_x, y, self.target_w, self.target_h)
        right_rect = pygame.Rect(right_x, y, self.target_w, self.target_h)
        return left_rect, right_rect

    def _generate_session_images(self, K: int):
        """Create K unique (left,right) panel surfaces for this session."""
        self.session_pairs = []
        inner_w = max(1, self.target_w - 12)
        inner_h = max(1, self.target_h - 34)
        for _ in range(K):
            left = self.image_factory.generate(self.rng, (inner_w, inner_h))
            right = self.image_factory.generate(self.rng, (inner_w, inner_h))
            self.session_pairs.append((left, right))

    def _load_pair_images(self, pair_idx: int):
        self.left_img = None
        self.right_img = None
        if 0 <= pair_idx < len(self.session_pairs):
            self.left_img, self.right_img = self.session_pairs[pair_idx]

    def _draw_panel_frames(self, surf: pygame.Surface):
        # Draw frames around the left/right panel rects
        pygame.draw.rect(surf, (255, 255, 255), self.left_rect, width=2, border_radius=6)
        pygame.draw.rect(surf, (255, 255, 255), self.right_rect, width=2, border_radius=6)

    def _blit_panel_images(self, surf: pygame.Surface):
        # Center the images in their frames
        if self.left_img is not None:
            x = self.left_rect.x + (self.left_rect.w - self.left_img.get_width()) // 2
            y = self.left_rect.y + (self.left_rect.h - self.left_img.get_height()) // 2
            surf.blit(self.left_img, (x, y))
        if self.right_img is not None:
            x = self.right_rect.x + (self.right_rect.w - self.right_img.get_width()) // 2
            y = self.right_rect.y + (self.right_rect.h - self.right_img.get_height()) // 2
            surf.blit(self.right_img, (x, y))

    def _prepare_trial(self):
        """Prepare trial state for the current trial_index based on episode_pair_schedule."""
        self.steps_in_trial = 0
        # Cursor starts near bottom-center
        self.cursor = np.array([self.W // 2, self.H - self.margin - 10], dtype=np.float32)

        # Pick which pair this trial uses
        self.curr_pair_idx = int(self.episode_pair_schedule[self.trial_index])
        # print("Current pair index:", self.curr_pair_idx, ", self.pair_probs=", self.pair_probs)

        # Determine probabilities for this pair
        if self.pair_probs is None:
            # If we didn't get fixed probs, generate them per session (length K)
            # Already generated in reset; just read them:
            self.curr_probs = self.session_pair_probs[self.curr_pair_idx]
        else:
            self.curr_probs = self.pair_probs[self.curr_pair_idx]

        # Optional labels; not strictly needed—kept for debug/info
        self.curr_labels = (0, 1)  # could be extended to actual IDs if desired

        # Images
        self._load_pair_images(self.curr_pair_idx)

        self.K = self.episode_pair_schedule[self.trial_index]

        # Randomize which side shows which target for this trial (50% flip)
        self.side_is_flipped = False
        if self.randomize_sides and (self.rng.random() < 0.5):
            self.side_is_flipped = True
            self.curr_probs = (self.curr_probs[1], self.curr_probs[0])
            self.curr_labels = (self.curr_labels[1], self.curr_labels[0])
            self.left_img, self.right_img = self.right_img, self.left_img
            self.flip_count_this_episode += 1

        self.obs_trial = False
        if self.trial_cond[self.trial_index] == "AO-o" or self.trial_cond[self.trial_index] == "OL-o":
            self.obs_trial = True

    # ---------- Gym API ----------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = _seeded_rng(seed)

        self._ensure_display()
        self.t = 0
        self.left_rect, self.right_rect = self._compute_target_rects()

        # Determine K for this session
        if self.pair_probs is None:
            K = self.session_K
            # Create fresh probabilities for this session (K pairs)
            # Draw two independent Bernoulli means for (left,right).
            # Optionally bias them a bit apart so the task isn't degenerate.
            # L = self.rng.uniform(0.1, 0.9, size=K)
            # R = self.rng.uniform(0.1, 0.9, size=K)
            L = np.ones(K) * 0.8
            R = np.ones(K) * 0.2
            # self.session_pair_probs = [(float(L[i]), float(R[i])) for i in range(K)]
            self.session_pair_probs = [
                (float(L[i]), float(R[i])) if np.random.rand() < 0.5 else (float(R[i]), float(L[i]))
                for i in range(K)
            ]

        else:
            K = len(self.pair_probs)
            self.session_pair_probs = list(self.pair_probs)

        # Fresh, unique images for this session only
        self.session_id += 1
        self._generate_session_images(K)
        self.image_labels = [0,1]




        # Build schedule: each pair index repeated N times, then shuffled
        #Set K = 3, Let K = 0 be IL, K=1 be AO, K=2 be OL
        # Total trials = N + 2*N + 2*N
        indices = np.empty(5*self.session_N, dtype=np.int32)
        self.trial_cond = ["IL"] * self.session_N + ["AO-o"] * (self.session_N) + ["AO-s"] * (self.session_N) + ["OL-o"] * (self.session_N) + ["OL-s"] * (self.session_N)
        trial_cond_perm = []
        for i in range(5*self.session_N):
            if i == 0:
                selection = random.choice(["IL", "AO-o", "AO-s", "OL-o", "OL-s"])
            else:
                choices = ["IL", "AO-o", "AO-s", "OL-o", "OL-s"]
                if(trial_cond_perm.count("IL") == self.session_N):
                    choices.remove("IL")
                if(trial_cond_perm.count("AO-o") == self.session_N or trial_cond_perm.count("AO-o") == trial_cond_perm.count("AO-s") + 1):
                    choices.remove("AO-o")
                if(trial_cond_perm.count("AO-s") == self.session_N or trial_cond_perm.count("AO-o") == trial_cond_perm.count("AO-s")):
                    choices.remove("AO-s")
                if(trial_cond_perm.count("OL-o") == self.session_N or trial_cond_perm.count("OL-o") == trial_cond_perm.count("OL-s") + 1):
                    choices.remove("OL-o")
                if(trial_cond_perm.count("OL-s") == self.session_N or trial_cond_perm.count("OL-o") == trial_cond_perm.count("OL-s")):    
                    choices.remove("OL-s")
                selection = random.choice(choices)
            trial_cond_perm.append(selection)
            if selection == "IL":
                indices[i] = 0
            elif selection == "AO-o" or selection == "AO-s":
                indices[i] = 1
            else:
                indices[i] = 2
        self.trial_cond = trial_cond_perm    
        self.episode_pair_schedule = indices.tolist()

        # if self.shuffle:
        #     self.episode_pair_schedule: List[int] = self.rng.permutation(indices).tolist()
        # else:
        #     self.episode_pair_schedule = indices.tolist()
        # self.trial_cond = self.trial_cond[indices]    
        # Trial bookkeeping
        self.trial_index = 0
        self.steps_in_trial = 0
        self.high_reward_choice_count = 0
        self.high_reward_choice_count_on_right = 0
        self.high_reward_choice_count_on_left = 0
        self.total_left_choices = 0
        self.total_right_choices = 0
        self.flip_count_this_episode = 0

        # --- NEW: per-session aggregates ---
        self.session_reward_sum = 0.0              # (4) cumulative rewards this session
        self.session_truncations = 0               # (1) cumulative trial timeouts this session
        self.session_terminations = 0              # (2) cumulative trial reaches this session


        # Prepare first trial
        self._prepare_trial()
        self.prev_cursor = self.cursor.copy()

        self.Q = np.zeros((3,2))  # Q-values for each condition and each side
        # self.alpha = 0.2886           # learning rate
        # self.tau = 0.3302              # softmax temperature
        

        # First frame
        frame = self._render()

        # print("[Env] Starting session", self.session_id, "with", K, "pairs repeated", self.session_N, "times each.")
        # print("[Env] Episode pair schedule:", self.episode_pair_schedule)
        # print("[Env] Pair probabilities:", self.session_pair_probs)

        info = {
            "pair_idx": self.curr_pair_idx,
            "target_probs": self.curr_probs,
            "labels": self.curr_labels,
            "current_K": self.K,
            "curr_trial_condition": self.trial_cond[self.trial_index],
            "trial_index": self.trial_index,
            "pair_index_in_session": self.curr_pair_idx,
            "total_trials_in_session": len(self.episode_pair_schedule),
            "high_reward_choice_count": self.high_reward_choice_count,
            "flips_this_episode": self.flip_count_this_episode,
            "session_id": self.session_id,
            "session_K": K,
            "session_N": self.session_N,
            "high_reward_choice_count": self.high_reward_choice_count,  # (alias for #3)
            "cum_session_high_reward_choices": self.high_reward_choice_count,  # (3)
            "cum_session_truncations": self.session_truncations,              # (1)
            "cum_session_terminations": self.session_terminations,            # (2)
            "cum_session_rewards": self.session_reward_sum,                    # (4)
        }
        return frame, info

    def step(self, action = None):
        # Action: (vx, vy) in [-1,1]
        a = np.array([0.0, 0.0], dtype=np.float32)

        if self.obs_trial and not self.expert_teacher and not self.unrealiable_teacher:
            # print("K=", self.K, "self.alpha:", self.alpha[self.K])
            # print("K=", self.K,"self.tau:", self.tau[self.K])
            q_left = self.Q[self.K,self.curr_labels[0]]
            q_right = self.Q[self.K,self.curr_labels[1]]
            prob_right = np.exp(q_right/self.tau[self.K]) / (np.exp(q_left/self.tau[self.K]) + np.exp(q_right/self.tau[self.K]))
            chosen_side = None
            if np.random.rand() < prob_right:
                chosen_side = "right"
            else:
                chosen_side = "left" 

            # print(f"[Env] Trial {self.trial_index} (K={self.K}, obs_trial): Q_left={q_left:.2f}, Q_right={q_right:.2f}, prob_right={prob_right:.2f}, chosen_side={chosen_side}, high_reward_side={'right' if self.curr_probs[1] > self.curr_probs[0] else 'left'}")   
            
            
            left_c  = np.array([self.left_rect.centerx,  self.left_rect.centery],  np.float32)
            right_c = np.array([self.right_rect.centerx, self.right_rect.centery], np.float32)

            # print("[Env] Trial ", self.trial_index, ", K=", self.K, ": Q_left=", np.round(q_left, 2), ", Q_right=", np.round(q_right, 2), ", prob_right=", np.round(prob_right, 2), ", chosen_side=", chosen_side)
            # print("Q-values:", self.Q)

            if chosen_side == "left":
                self.cursor = left_c
            else:
                self.cursor = right_c
        elif self.obs_trial and self.expert_teacher:
            # print(f"[Env] Trial {self.trial_index} (K={self.K}, obs_trial): expert teacher mode")
            high_side = "L" if self.curr_probs[0] >= self.curr_probs[1] else "R"
            chosen_side = high_side if np.random.rand() < self.eps else ("R" if high_side == "R" else "L")
            left_c  = np.array([self.left_rect.centerx,  self.left_rect.centery],  np.float32)
            right_c = np.array([self.right_rect.centerx, self.right_rect.centery], np.float32)

            if chosen_side in ("L","left"):
                self.cursor = left_c
            else:
                self.cursor = right_c

        elif self.obs_trial and self.unrealiable_teacher:
            # print(f"[Env] Trial {self.trial_index} (K={self.K}, obs_trial): unreliable teacher mode")
            chosen_side = "L" if np.random.rand() < 0.5 else "R"  # random choice, ignoring probabilities    
            left_c  = np.array([self.left_rect.centerx,  self.left_rect.centery],  np.float32)
            right_c = np.array([self.right_rect.centerx, self.right_rect.centery], np.float32)

            if chosen_side in ("L","left"):
                self.cursor = left_c
            else:
                self.cursor = right_c

        else:  
            # print(f"[Env] Trial {self.trial_index} (K={self.K}): regular trial with action input")
            a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            # Scale velocity to pixels/frame
            speed_px = 12.0
            self.cursor += a * speed_px
            self.cursor[0] = float(np.clip(self.cursor[0], 0, self.W - 1))
            self.cursor[1] = float(np.clip(self.cursor[1], 0, self.H - 1))

        self.steps_in_trial += 1
        trial_ended = False
        reached = False
        reward = 0.0
        selected_high = False
        timeout = False
        high_side = "L" if self.curr_probs[0] >= self.curr_probs[1] else "R"

        # --- NEW: truncate the trial if we touched any boundary ---
        hit_boundary = (
            self.cursor[0] <= 0.0 or self.cursor[0] >= self.W - 1 or
            self.cursor[1] <= 0.0 or self.cursor[1] >= self.H - 1
        )
        if hit_boundary and not trial_ended:
            trial_ended = True
            timeout = True      # count as a truncation in session stats
            reward = 0.0        # (keep at 0 unless you want to penalize)
            reached = False


        # Check for reaching left/right panel
        pt = (int(self.cursor[0]), int(self.cursor[1]))
        chose_side = None
        if self.left_rect.collidepoint(pt):
            chose_side = 0
            self.total_left_choices += 1
        elif self.right_rect.collidepoint(pt):
            chose_side = 1
            self.total_right_choices += 1

        if chose_side is not None:
            reached = True
            p = float(self.curr_probs[chose_side])
            reward = 1.0 if (self.rng.random() < p) else 0.0
            # print(f"[Env] Trial {self.trial_index}: reached side {'L' if chose_side==0 else 'R'} with p={p:.2f}, reward={reward}")
            selected_high = (chose_side == (0 if self.curr_probs[0] >= self.curr_probs[1] else 1))
            if selected_high:
                self.high_reward_choice_count += 1
                self.high_reward_choice_count_on_left += (1 if chose_side == 0 else 0)
                self.high_reward_choice_count_on_right += (1 if chose_side == 1 else 0)
            trial_ended = True
            if self.obs_trial and not self.expert_teacher and not self.unrealiable_teacher:
                chosen_label = self.curr_labels[chose_side]         # label currently shown on chosen side
                other_label = self.curr_labels[1 - chose_side]      # label on the other side

                self.Q[self.K, chosen_label] += self.alpha[self.K] * (
                    reward - self.Q[self.K, chosen_label]
                )
                # only keep this if you really want complementary values
                # self.Q[self.K, other_label] = 1.0 - self.Q[self.K, chosen_label]  # keep the other side's Q complementary for simplicity

        # Timeout
        if not trial_ended and self.steps_in_trial >= self.steps_per_trial:
            trial_ended = True
            timeout = True

        # If the trial ended, advance schedule or end episode
        prev_pair_idx = int(self.curr_pair_idx)
        prev_trial_idx = int(self.trial_index)

        terminated = False
        if trial_ended:
            print("Trial = ", self.trial_index, ", reached:", reached, ", Condition = ", self.trial_cond[self.trial_index], ", K = ", self.K, ", obs_trial=", self.obs_trial, ", selected_side=", chose_side, ", selected_high_reward=", selected_high, ", reward=", reward)
            old_pair_idx = self.curr_pair_idx
            # print("End trial = ", self.trial_index, ", pair_index_in_session=", old_pair_idx)
            if self.trial_cond == "AO-o":
                reward = -1

            self.session_reward_sum += float(reward)
            if timeout:
                self.session_truncations += 1      # (1)
            else:
                self.session_terminations += 1     # (2) ended by reaching a target

            self.trial_index += 1
            if self.trial_index >= len(self.episode_pair_schedule):
                terminated = True  # end of session
                self.trial_index -= 1  # stay at last trial for info
            else:          
                self._prepare_trial()
        frame = self._render()

        dxy = self.cursor - self.prev_cursor
        self.prev_cursor = self.cursor.copy()

        curr_pair_idx = int(self.curr_pair_idx)

        info = {
            "reached_target": reached,
            "curr_trial_condition": self.trial_cond[self.trial_index],  ## To read during trial
            "prev_trial_condition": self.trial_cond[prev_trial_idx],   ## To read after trial end
            "hit_boundary": bool(hit_boundary),   # <--- NEW, for debugging/analysis
            "total_trials_in_session": len(self.episode_pair_schedule),
            "trial_index": self.trial_index,
            "pair_index_in_session": curr_pair_idx,               # current pair (for returned frame)
            "prev_trial_index": prev_trial_idx,                   # previous (the one that just ended)
            "prev_pair_index_in_session": prev_pair_idx,          # previous (the one that just ended)
            "trial_ended": trial_ended,
            "timeout": timeout,
            "selected_target" : chose_side,
            "side_is_flipped": getattr(self, "side_is_flipped", False),
            "high_reward_choice_count": self.high_reward_choice_count,
            "high_reward_choice_count_on_left": self.high_reward_choice_count_on_left,
            "high_reward_choice_count_on_right": self.high_reward_choice_count_on_right,
            "total_left_choices": self.total_left_choices,
            "total_right_choices": self.total_right_choices,
            "selected_high_reward_this_trial": bool(selected_high),
            "high_prob_side": high_side,
            "flips_this_episode": getattr(self, "flip_count_this_episode", 0),
            "session_id": self.session_id,
            "probs_this_trial": self.curr_probs,
            # --- NEW: cumulative per-session stats you asked for ---
            "cum_session_truncations": self.session_truncations,                   # (1)
            "cum_session_terminations": self.session_terminations,                 # (2)
            "cum_session_high_reward_choices": self.high_reward_choice_count,      # (3)
            "cum_session_rewards": self.session_reward_sum,                         # (4)
            "cursor_xy": (float(self.cursor[0]), float(self.cursor[1])),
            "cursor_dxy": (float(dxy[0]), float(dxy[1])),
            "left_center":  (int(self.left_rect.centerx),  int(self.left_rect.centery)),
            "right_center": (int(self.right_rect.centerx), int(self.right_rect.centery)),
            "arena_wh": (self.W, self.H),


        }
        truncated = False
        return frame, float(reward), terminated, truncated, info

    # ---------- rendering ----------

    def _render(self) -> np.ndarray:
        surf = self._backbuffer
        surf.fill((30, 30, 30))

        # Draw panels and images
        self._draw_panel_frames(surf)
        self._blit_panel_images(surf)

        # Cursor
        pygame.draw.circle(surf, (255, 255, 255), (int(self.cursor[0]), int(self.cursor[1])), 5)

        # HUD
        text = self._font.render(
            f"session {self.session_id}| episode {self.trial_index+1}/{len(self.episode_pair_schedule)}",
            True,
            (200, 200, 200),
        )
        surf.blit(text, (8, self.H - 22))

        # Push to window if human
        if self.render_mode == "human" and self._window is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass
            self._window.blit(surf, (0, 0))
            pygame.display.flip()

        # Return rgb array
        arr = pygame.surfarray.array3d(surf)  # (W,H,3)
        arr = np.transpose(arr, (1, 0, 2))    # -> (H,W,3)
        return arr

    def render(self):
        if self.render_mode == "human":
            # No-op; human frames are already displayed in _render()
            pass
        else:
            return self._render()

    def close(self):
        if self._window is not None:
            try:
                pygame.display.quit()
            except Exception:
                pass
            self._window = None
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    # Example: K=3 pairs × N=4 repeats = 12 trials per session; run 3 sessions.
    env = TwoChoiceReachingEnv(
        W = 384,
        H = 400,
        render_mode="human",
        seed=0,
        session_K=3,
        session_N=5,
        trial_ms=60000,
        randomize_sides=True,
    )

    num_sessions = 1
    for ep in range(num_sessions):
        obs, info = env.reset()
        # print("obs type:", type(obs))
        done = False
        ep_reward = 0.0
        while not done:
            # random policy: small bias upward
            a = np.array([np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, -0.1)], dtype=np.float32)
            obs, r, term, trunc, info = env.step(a)
            # print("a:", a)
            ep_reward += r
            done = term or trunc
            # print(f"probs={info.get('probs_this_trial', (0.0,0.0))}, reward={r:.1f}, total_ep_reward={ep_reward:.1f}")
        print(
            f"Session {ep+1}: reward_sum={info['cum_session_rewards']:.1f}, "
            f"truncs={info['cum_session_truncations']}, "
            f"terms={info['cum_session_terminations']}, "
            f"trials={info['total_trials_in_session']}, "
            f"high_reward_choices={info['cum_session_high_reward_choices']}, "
            f"flips={info.get('flips_this_episode', 0)}",
            f"trial_count={info['trial_index']}",
            f"curr_pair_idx={info['pair_index_in_session']}",
        )
    env.close()