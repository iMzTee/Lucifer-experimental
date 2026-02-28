"""game_state.py — TensorState: all environment state as contiguous GPU tensors.

Shape convention: (E, ...) where E = n_envs.
Cars: (E, 4, ...) for 4 players in 2v2.
All tensors live on the same device (typically 'cuda').
"""

import torch
from .constants import N_BOOST_PADS


class TensorState:
    """All environment state as contiguous GPU tensors.

    Memory layout per env: ~1.5 KB → 50,000 envs = 75 MB.
    """

    def __init__(self, n_envs, device='cuda'):
        E = n_envs
        self.n_envs = E
        self.n_agents = 4  # 2v2
        self.device = device

        # ── Ball state ──
        self.ball_pos = torch.zeros(E, 3, device=device)
        self.ball_vel = torch.zeros(E, 3, device=device)
        self.ball_ang_vel = torch.zeros(E, 3, device=device)

        # ── Car state ──
        self.car_pos = torch.zeros(E, 4, 3, device=device)
        self.car_vel = torch.zeros(E, 4, 3, device=device)
        self.car_ang_vel = torch.zeros(E, 4, 3, device=device)
        self.car_quat = torch.zeros(E, 4, 4, device=device)  # (w, x, y, z)
        # Initialize quaternions to identity (w=1)
        self.car_quat[:, :, 0] = 1.0

        # Cached rotation vectors (derived from quaternion)
        self.car_fwd = torch.zeros(E, 4, 3, device=device)
        self.car_fwd[:, :, 0] = 1.0  # default forward = +x
        self.car_up = torch.zeros(E, 4, 3, device=device)
        self.car_up[:, :, 2] = 1.0   # default up = +z

        # ── Car scalars ──
        self.car_boost = torch.zeros(E, 4, device=device)
        self.car_on_ground = torch.ones(E, 4, device=device)    # start grounded
        self.car_has_flip = torch.ones(E, 4, device=device)     # start with flip
        self.car_is_demoed = torch.zeros(E, 4, device=device)
        self.car_demoed_timer = torch.zeros(E, 4, device=device)
        self.car_jump_timer = torch.zeros(E, 4, device=device)
        self.car_is_jumping = torch.zeros(E, 4, device=device)
        self.car_has_jumped = torch.zeros(E, 4, device=device)
        self.car_has_flipped = torch.zeros(E, 4, device=device)
        self.car_ball_touched = torch.zeros(E, 4, device=device)

        # Team assignment: 0=blue, 1=orange → [0, 0, 1, 1]
        self.car_team = torch.zeros(E, 4, dtype=torch.long, device=device)
        self.car_team[:, 2] = 1
        self.car_team[:, 3] = 1

        # ── Boost pads ──
        # Timer = 0 means available; >0 means respawning (counts down)
        self.boost_pad_timers = torch.zeros(E, N_BOOST_PADS, device=device)

        # ── Scores ──
        self.blue_score = torch.zeros(E, dtype=torch.long, device=device)
        self.orange_score = torch.zeros(E, dtype=torch.long, device=device)

        # ── Episode tracking ──
        self.step_count = torch.zeros(E, dtype=torch.long, device=device)

        # ── Match event counters (cumulative per episode) ──
        self.match_goals = torch.zeros(E, 4, device=device)
        self.match_saves = torch.zeros(E, 4, device=device)
        self.match_shots = torch.zeros(E, 4, device=device)
        self.match_demos = torch.zeros(E, 4, device=device)

        # ── Previous ball speed (for ball acceleration reward) ──
        self.prev_ball_speed = torch.zeros(E, device=device)

    def clone(self):
        """Create a deep copy of this state (for snapshot/comparison)."""
        new = TensorState.__new__(TensorState)
        new.n_envs = self.n_envs
        new.n_agents = self.n_agents
        new.device = self.device
        for attr in self.__dict__:
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new, attr, val.clone())
            else:
                setattr(new, attr, val)
        return new
