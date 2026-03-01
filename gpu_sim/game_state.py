"""game_state.py — TensorState: all environment state as contiguous GPU tensors.

Shape convention: (E, ...) where E = n_envs.
Cars: (E, A, ...) where A = n_agents (1, 2, or 4).
All tensors live on the same device (typically 'cuda').
"""

import torch
from .constants import N_BOOST_PADS, get_agent_layout


class TensorState:
    """All environment state as contiguous GPU tensors.

    Supports variable n_agents: 1 (1v0), 2 (1v1), or 4 (2v2).
    """

    def __init__(self, n_envs, device='cuda', n_agents=4):
        E = n_envs
        A = n_agents
        self.n_envs = E
        self.n_agents = A
        self.device = device

        # Agent layout
        layout = get_agent_layout(A)
        self.layout = layout

        # ── Ball state ──
        self.ball_pos = torch.zeros(E, 3, device=device)
        self.ball_vel = torch.zeros(E, 3, device=device)
        self.ball_ang_vel = torch.zeros(E, 3, device=device)

        # ── Car state ──
        self.car_pos = torch.zeros(E, A, 3, device=device)
        self.car_vel = torch.zeros(E, A, 3, device=device)
        self.car_ang_vel = torch.zeros(E, A, 3, device=device)
        self.car_quat = torch.zeros(E, A, 4, device=device)  # (w, x, y, z)
        self.car_quat[:, :, 0] = 1.0  # identity

        # Cached rotation vectors (derived from quaternion)
        self.car_fwd = torch.zeros(E, A, 3, device=device)
        self.car_fwd[:, :, 0] = 1.0  # default forward = +x
        self.car_up = torch.zeros(E, A, 3, device=device)
        self.car_up[:, :, 2] = 1.0   # default up = +z

        # ── Car scalars ──
        self.car_boost = torch.zeros(E, A, device=device)
        self.car_on_ground = torch.ones(E, A, device=device)
        self.car_has_flip = torch.ones(E, A, device=device)
        self.car_is_demoed = torch.zeros(E, A, device=device)
        self.car_demoed_timer = torch.zeros(E, A, device=device)
        self.car_jump_timer = torch.zeros(E, A, device=device)
        self.car_is_jumping = torch.zeros(E, A, device=device)
        self.car_has_jumped = torch.zeros(E, A, device=device)
        self.car_has_flipped = torch.zeros(E, A, device=device)
        self.car_ball_touched = torch.zeros(E, A, device=device)

        # Surface normal for wall/ceiling driving (default = floor up)
        self.car_surface_normal = torch.zeros(E, A, 3, device=device)
        self.car_surface_normal[:, :, 2] = 1.0

        # Handbrake state (set by physics, read by arena friction)
        self.car_handbrake = torch.zeros(E, A, device=device)

        # Team assignment from layout
        self.car_team = torch.tensor(
            layout["car_team"], dtype=torch.long, device=device
        ).unsqueeze(0).expand(E, -1).clone()

        # ── Boost pads ──
        self.boost_pad_timers = torch.zeros(E, N_BOOST_PADS, device=device)

        # ── Scores ──
        self.blue_score = torch.zeros(E, dtype=torch.long, device=device)
        self.orange_score = torch.zeros(E, dtype=torch.long, device=device)

        # ── Episode tracking ──
        self.step_count = torch.zeros(E, dtype=torch.long, device=device)

        # ── Match event counters ──
        self.match_goals = torch.zeros(E, A, device=device)
        self.match_saves = torch.zeros(E, A, device=device)
        self.match_shots = torch.zeros(E, A, device=device)
        self.match_demos = torch.zeros(E, A, device=device)

        # ── Previous ball speed (for ball acceleration reward) ──
        self.prev_ball_speed = torch.zeros(E, device=device)

    def clone(self):
        """Create a deep copy of this state."""
        new = TensorState.__new__(TensorState)
        new.n_envs = self.n_envs
        new.n_agents = self.n_agents
        new.device = self.device
        new.layout = self.layout
        for attr in self.__dict__:
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(new, attr, val.clone())
            elif attr not in ('n_envs', 'n_agents', 'device', 'layout'):
                setattr(new, attr, val)
        return new
