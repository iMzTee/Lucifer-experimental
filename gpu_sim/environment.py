"""environment.py — GPUEnvironment: step, reset, terminal detection.

Orchestrates physics ticks, collision detection, boost pad pickup,
and episode management. Supports variable n_agents (1v0, 1v1, 2v2)
with mechanic-specific training scenarios.
"""

import torch
import math
from .game_state import TensorState
from .constants import (
    ARENA_HALF_X, ARENA_HALF_Y, ARENA_HEIGHT,
    BALL_RADIUS, GOAL_HALF_WIDTH, GOAL_HEIGHT, BACK_NET_Y,
    BOOST_PAD_POSITIONS, N_BOOST_PADS, N_LARGE_PADS,
    LARGE_PAD_BOOST, SMALL_PAD_BOOST, LARGE_PAD_RESPAWN, SMALL_PAD_RESPAWN,
    BOOST_PAD_PICKUP_RADIUS, CAR_MAX_SPEED, DT, PHYSICS_HZ,
    KICKOFF_POSITIONS, STAGE_CONFIG, get_agent_layout,
)
from .physics import (
    apply_car_controls, integrate_positions, update_rotation_vectors,
    update_demoed_cars,
)
from .arena import arena_collide_ball, arena_collide_cars
from .collision import ball_car_collision, car_car_collision
from .utils import quat_from_euler, quat_to_fwd_up


class GPUEnvironment:
    """GPU-accelerated Rocket League environment for parallel simulation.

    Supports 1v0 (n_agents=1), 1v1 (n_agents=2), 2v2 (n_agents=4).
    """

    def __init__(self, n_envs, device='cuda', stage=0):
        self.n_envs = n_envs
        self.device = device
        self.stage = stage

        cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[0])
        self.tick_skip = cfg["tick_skip"]
        self.timeout = cfg["timeout"]
        n_agents = cfg.get("n_agents", 4)
        self.n_agents = n_agents
        self.layout = get_agent_layout(n_agents)

        self.state = TensorState(n_envs, device, n_agents=n_agents)

        # Pre-move boost pad positions to device
        self._pad_pos = BOOST_PAD_POSITIONS.to(device)
        self._pad_pos_xy = self._pad_pos[:, :2]

        # Boost pad respawn times
        self._pad_respawn = torch.zeros(N_BOOST_PADS, device=device)
        self._pad_respawn[:N_LARGE_PADS] = LARGE_PAD_RESPAWN
        self._pad_respawn[N_LARGE_PADS:] = SMALL_PAD_RESPAWN

        # Boost pad amounts
        self._pad_amount = torch.zeros(N_BOOST_PADS, device=device)
        self._pad_amount[:N_LARGE_PADS] = LARGE_PAD_BOOST
        self._pad_amount[N_LARGE_PADS:] = SMALL_PAD_BOOST

        # Kickoff positions on device
        self._kickoff_pos = KICKOFF_POSITIONS.to(device)

        # Goal tracking
        self._prev_blue_score = torch.zeros(n_envs, dtype=torch.long, device=device)
        self._prev_orange_score = torch.zeros(n_envs, dtype=torch.long, device=device)

        self.reset_all()

    def reset_all(self):
        """Reset all environments to initial state."""
        mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        self._reset_envs(mask)
        self._prev_blue_score[:] = 0
        self._prev_orange_score[:] = 0

    def _reset_envs(self, mask):
        """Reset specific environments with stage-appropriate scenarios."""
        if not mask.any():
            return

        n_reset = mask.sum().item()
        s = self.state
        A = self.n_agents

        # ── Ball: center at resting height ──
        s.ball_pos[mask] = torch.tensor([0.0, 0.0, BALL_RADIUS + 1.0], device=self.device)
        s.ball_vel[mask] = 0.0
        s.ball_ang_vel[mask] = 0.0

        # ── Scenario selection by agent count ──
        if A == 1:
            self._reset_stage0_scenarios(mask, n_reset)
        elif A == 2:
            self._reset_stage1_2_scenarios(mask, n_reset)
        else:
            self._reset_stage3_scenarios(mask, n_reset)

        # ── Reset common state ──
        s.car_boost[mask] = 0.33
        s.car_on_ground[mask] = 1.0
        s.car_has_flip[mask] = 1.0
        s.car_is_demoed[mask] = 0.0
        s.car_demoed_timer[mask] = 0.0
        s.car_has_jumped[mask] = 0.0
        s.car_has_flipped[mask] = 0.0
        s.car_is_jumping[mask] = 0.0
        s.car_jump_timer[mask] = 0.0
        s.car_ball_touched[mask] = 0.0
        s.car_vel[mask] = 0.0
        s.car_ang_vel[mask] = 0.0
        s.car_surface_normal[mask] = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Reset scores
        s.blue_score[mask] = 0
        s.orange_score[mask] = 0
        s.step_count[mask] = 0

        # Reset event counters
        s.match_goals[mask] = 0.0
        s.match_saves[mask] = 0.0
        s.match_shots[mask] = 0.0
        s.match_demos[mask] = 0.0

        # Reset boost pads
        s.boost_pad_timers[mask] = 0.0

        # Reset tracking
        s.prev_ball_speed[mask] = 0.0

        # Score tracking
        self._prev_blue_score[mask] = 0
        self._prev_orange_score[mask] = 0

        # Update rotation vectors
        fwd, up = quat_to_fwd_up(s.car_quat)
        s.car_fwd = fwd
        s.car_up = up

    # ═══════════════════════════════════════════════════
    # Stage 0: 1v0 — 5 ground-only scenarios
    # ═══════════════════════════════════════════════════

    def _reset_stage0_scenarios(self, mask, n):
        """Stage 0 (1v0): ground-only mechanic-focused scenarios."""
        r = torch.rand(n, device=self.device)
        indices = mask.nonzero(as_tuple=True)[0]

        # Scenario distribution (all ground-only):
        # Kickoff 25%, SpeedFlip 20%, GroundDribble 20%, AerialTouch 15%, FreePlay 20%
        thresholds = [0.25, 0.45, 0.65, 0.80, 1.00]
        scenarios = [
            self._reset_kickoff_1v0,
            self._reset_speed_flip_drill,
            self._reset_ground_dribble,
            self._reset_aerial_touch,
            self._reset_free_play_1v0,
        ]

        for scenario_idx in range(5):
            lo = 0.0 if scenario_idx == 0 else thresholds[scenario_idx - 1]
            hi = thresholds[scenario_idx]
            local_mask = (r >= lo) & (r < hi)
            if not local_mask.any():
                continue
            full_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            full_mask[indices[local_mask]] = True
            n_s = full_mask.sum().item()
            scenarios[scenario_idx](full_mask, n_s)

    def _reset_kickoff_1v0(self, mask, n):
        """Solo kickoff: random kickoff position, ball at center."""
        s = self.state
        pos_idx = torch.randint(0, 5, (n,), device=self.device)
        positions = self._kickoff_pos[pos_idx]
        s.car_pos[mask, 0] = positions
        yaw = torch.full((n,), math.pi / 2, device=self.device)
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = 0.33

    def _reset_speed_flip_drill(self, mask, n):
        """Speed flip practice: far end, facing ball at center, 33% boost."""
        s = self.state
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 1000 - 500
        pos[:, 1] = -(torch.rand(n, device=self.device) * 500 + 4000)  # -4000 to -4500
        pos[:, 2] = 17.0
        s.car_pos[mask, 0] = pos
        yaw = torch.full((n,), math.pi / 2, device=self.device)
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = 0.33

    def _reset_wave_dash_drill(self, mask, n):
        """Wave dash practice: car slightly in air, moving forward, has flip."""
        s = self.state
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 2] = torch.rand(n, device=self.device) * 100 + 50  # 50-150
        s.car_pos[mask, 0] = pos

        # Moving forward at 500-1500 uu/s
        yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))

        # Set velocity in forward direction
        fwd_speed = torch.rand(n, device=self.device) * 1000 + 500
        vel = torch.zeros(n, 3, device=self.device)
        vel[:, 0] = fwd_speed * torch.cos(yaw)
        vel[:, 1] = fwd_speed * torch.sin(yaw)
        s.car_vel[mask, 0] = vel

        s.car_on_ground[mask, 0] = 0.0
        s.car_has_jumped[mask, 0] = 1.0
        s.car_has_flip[mask, 0] = 1.0
        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.5

        # Ball out of the way
        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        ball_pos[:, 2] = BALL_RADIUS + 1.0
        s.ball_pos[mask] = ball_pos

    def _reset_ground_dribble(self, mask, n):
        """Ground dribble: ball on top of car, moving forward slowly."""
        s = self.state
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 2] = 17.0
        s.car_pos[mask, 0] = pos

        yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))

        # Ball on top of car (slightly ahead and above)
        ball_pos = pos.clone()
        ball_pos[:, 0] += 20.0 * torch.cos(yaw)
        ball_pos[:, 1] += 20.0 * torch.sin(yaw)
        ball_pos[:, 2] = 150.0  # on car hood
        s.ball_pos[mask] = ball_pos

        # Slow forward ball velocity
        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = 200.0 * torch.cos(yaw)
        ball_vel[:, 1] = 200.0 * torch.sin(yaw)
        s.ball_vel[mask] = ball_vel

        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.5 + 0.3

    def _reset_wall_drive(self, mask, n):
        """Wall drive practice: car on wall, ball nearby on wall or in air."""
        s = self.state

        # Random wall: 0=right(+X), 1=left(-X), 2=orange(+Y), 3=blue(-Y)
        wall_choice = torch.randint(0, 4, (n,), device=self.device)

        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 2] = torch.rand(n, device=self.device) * 600 + 200  # 200-800 height

        # Position on wall surface
        right = wall_choice == 0
        left = wall_choice == 1
        orange = wall_choice == 2
        blue = wall_choice == 3

        pos[:, 0] = torch.where(right, torch.tensor(ARENA_HALF_X - 50.0, device=self.device),
                    torch.where(left, torch.tensor(-(ARENA_HALF_X - 50.0), device=self.device),
                    torch.rand(n, device=self.device) * 4000 - 2000))
        pos[:, 1] = torch.where(orange, torch.tensor(ARENA_HALF_Y - 50.0, device=self.device),
                    torch.where(blue, torch.tensor(-(ARENA_HALF_Y - 50.0), device=self.device),
                    torch.rand(n, device=self.device) * 6000 - 3000))

        s.car_pos[mask, 0] = pos

        # Orient car along wall (facing up the wall)
        pitch = torch.zeros(n, device=self.device)
        yaw = torch.zeros(n, device=self.device)
        roll = torch.zeros(n, device=self.device)

        # On side walls: roll ±90°; on back walls: pitch ±90°
        roll = torch.where(right, torch.tensor(-math.pi / 2, device=self.device), roll)
        roll = torch.where(left, torch.tensor(math.pi / 2, device=self.device), roll)
        pitch = torch.where(orange, torch.tensor(-math.pi / 2, device=self.device), pitch)
        pitch = torch.where(blue, torch.tensor(math.pi / 2, device=self.device), pitch)

        s.car_quat[mask, 0] = quat_from_euler(pitch, yaw, roll)

        # Car is on wall surface
        s.car_on_ground[mask, 0] = 1.0
        # Surface normals
        normals = torch.zeros(n, 3, device=self.device)
        normals[:, 0] = torch.where(right, torch.tensor(-1.0, device=self.device),
                        torch.where(left, torch.tensor(1.0, device=self.device),
                        torch.tensor(0.0, device=self.device)))
        normals[:, 1] = torch.where(orange, torch.tensor(-1.0, device=self.device),
                        torch.where(blue, torch.tensor(1.0, device=self.device),
                        torch.tensor(0.0, device=self.device)))
        s.car_surface_normal[mask, 0] = normals

        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.5 + 0.5

        # Ball near car on wall or in air
        ball_pos = pos.clone()
        ball_pos[:, 2] += torch.rand(n, device=self.device) * 300
        # Push ball slightly away from wall
        ball_pos[:, 0] += normals[:, 0] * (BALL_RADIUS + 50)
        ball_pos[:, 1] += normals[:, 1] * (BALL_RADIUS + 50)
        s.ball_pos[mask] = ball_pos

    def _reset_aerial_touch(self, mask, n):
        """Aerial practice: car on ground under high ball, 70%+ boost."""
        s = self.state
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 2] = 17.0
        s.car_pos[mask, 0] = pos

        # Ball high in air above car
        ball_pos = pos.clone()
        ball_pos[:, 0] += torch.rand(n, device=self.device) * 400 - 200
        ball_pos[:, 1] += torch.rand(n, device=self.device) * 400 - 200
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 1200 + 400  # 400-1600
        s.ball_pos[mask] = ball_pos

        # Slow ball drift
        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 400 - 200
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 400 - 200
        ball_vel[:, 2] = torch.rand(n, device=self.device) * 200 - 100
        s.ball_vel[mask] = ball_vel

        # Face ball
        to_ball = ball_pos[:, :2] - pos[:, :2]
        yaw = torch.atan2(to_ball[:, 1], to_ball[:, 0])
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))

        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.3 + 0.7  # 70-100%

    def _reset_free_play_1v0(self, mask, n):
        """Free play: random position and ball."""
        s = self.state
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 7000 - 3500
        pos[:, 1] = torch.rand(n, device=self.device) * 9000 - 4500
        pos[:, 2] = 17.0
        s.car_pos[mask, 0] = pos

        yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = torch.rand(n, device=self.device)

        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 8000 - 4000
        ball_pos[:, 2] = BALL_RADIUS + 1.0
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 3000 - 1500
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 3000 - 1500
        s.ball_vel[mask] = ball_vel

    # ═══════════════════════════════════════════════════
    # Stage 1-2: 1v1 — 8 mechanic-specific scenarios
    # ═══════════════════════════════════════════════════

    def _reset_stage1_2_scenarios(self, mask, n):
        """Stage 1-2 (1v1): mechanic-focused scenarios."""
        r = torch.rand(n, device=self.device)
        indices = mask.nonzero(as_tuple=True)[0]

        # Kickoff 20%, AirDribble 10%, FlipReset 10%, GroundPlay 15%,
        # AerialChallenge 10%, WallPlay 10%, Shooting 15%, Saving 10%
        thresholds = [0.20, 0.30, 0.40, 0.55, 0.65, 0.75, 0.90, 1.00]

        scenarios = [
            self._reset_kickoff_1v1,
            self._reset_air_dribble_setup,
            self._reset_flip_reset_setup,
            self._reset_ground_play_1v1,
            self._reset_aerial_challenge,
            self._reset_wall_play_1v1,
            self._reset_shooting_practice,
            self._reset_saving_practice,
        ]

        for idx, scenario_fn in enumerate(scenarios):
            lo = 0.0 if idx == 0 else thresholds[idx - 1]
            hi = thresholds[idx]
            local_mask = (r >= lo) & (r < hi)
            if not local_mask.any():
                continue
            full_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            full_mask[indices[local_mask]] = True
            n_s = full_mask.sum().item()
            scenario_fn(full_mask, n_s)

    def _reset_kickoff_1v1(self, mask, n):
        """1v1 kickoff: both at kickoff positions."""
        s = self.state
        # Blue
        pos_idx = torch.randint(0, 5, (n,), device=self.device)
        s.car_pos[mask, 0] = self._kickoff_pos[pos_idx]
        yaw = torch.full((n,), math.pi / 2, device=self.device)
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = 0.33

        # Orange (mirrored)
        pos_idx = torch.randint(0, 5, (n,), device=self.device)
        positions = self._kickoff_pos[pos_idx] * torch.tensor([-1.0, -1.0, 1.0], device=self.device)
        s.car_pos[mask, 1] = positions
        yaw = torch.full((n,), -math.pi / 2, device=self.device)
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = 0.33

    def _reset_air_dribble_setup(self, mask, n):
        """Air dribble setup: blue on wall, ball rolling up wall ahead."""
        s = self.state

        # Blue car on right wall, mid-height
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = ARENA_HALF_X - 50.0
        pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 2] = torch.rand(n, device=self.device) * 400 + 300  # 300-700
        s.car_pos[mask, 0] = pos

        # Car oriented along wall, facing up
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device))
        s.car_on_ground[mask, 0] = 1.0
        s.car_surface_normal[mask, 0] = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.2 + 0.8  # 80-100%

        # Ball on wall ahead, slightly above car
        ball_pos = pos.clone()
        ball_pos[:, 2] += torch.rand(n, device=self.device) * 200 + 100
        ball_pos[:, 0] = ARENA_HALF_X - BALL_RADIUS - 10  # on the wall
        s.ball_pos[mask] = ball_pos

        # Ball rolling up the wall
        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 2] = torch.rand(n, device=self.device) * 500 + 300
        s.ball_vel[mask] = ball_vel

        # Orange opponent on ground, far side
        pos_o = torch.zeros(n, 3, device=self.device)
        pos_o[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos_o[:, 1] = torch.rand(n, device=self.device) * 2000 + 2000
        pos_o[:, 2] = 17.0
        s.car_pos[mask, 1] = pos_o
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = torch.rand(n, device=self.device)

    def _reset_flip_reset_setup(self, mask, n):
        """Flip reset setup: blue in air under ball, upside-down approach."""
        s = self.state

        # Ball high in air
        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 3000 - 1500
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 3000 - 1500
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 800 + 600  # 600-1400
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 300 - 150
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 300 - 150
        ball_vel[:, 2] = torch.rand(n, device=self.device) * 200 - 100
        s.ball_vel[mask] = ball_vel

        # Blue car below ball, approaching from underneath
        car_pos = ball_pos.clone()
        car_pos[:, 2] -= torch.rand(n, device=self.device) * 200 + 100  # 100-300 below
        car_pos[:, 0] += torch.rand(n, device=self.device) * 200 - 100
        s.car_pos[mask, 0] = car_pos

        # Upside-down orientation (approaching ball from below)
        s.car_quat[mask, 0] = quat_from_euler(
            torch.full((n,), math.pi, device=self.device),  # flipped
            torch.rand(n, device=self.device) * 0.5 - 0.25,
            torch.zeros(n, device=self.device))
        s.car_on_ground[mask, 0] = 0.0
        s.car_has_jumped[mask, 0] = 1.0
        s.car_has_flip[mask, 0] = 0.0  # no flip yet — need reset from ball
        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.3 + 0.5

        # Upward velocity to reach ball
        vel = torch.zeros(n, 3, device=self.device)
        vel[:, 2] = torch.rand(n, device=self.device) * 300 + 200
        s.car_vel[mask, 0] = vel

        # Orange on ground
        pos_o = torch.zeros(n, 3, device=self.device)
        pos_o[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos_o[:, 1] = torch.rand(n, device=self.device) * 2000 + 2000
        pos_o[:, 2] = 17.0
        s.car_pos[mask, 1] = pos_o
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = torch.rand(n, device=self.device)

    def _reset_ground_play_1v1(self, mask, n):
        """General 1v1 ground play."""
        s = self.state

        # Ball at random ground position
        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 8000 - 4000
        ball_pos[:, 2] = BALL_RADIUS + 1.0
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 3000 - 1500
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 3000 - 1500
        s.ball_vel[mask] = ball_vel

        # Blue in blue half
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 7000 - 3500
        pos[:, 1] = torch.rand(n, device=self.device) * 4500 - 4500
        pos[:, 2] = 17.0
        s.car_pos[mask, 0] = pos
        yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = torch.rand(n, device=self.device)

        # Orange in orange half
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 7000 - 3500
        pos[:, 1] = torch.rand(n, device=self.device) * 4500
        pos[:, 2] = 17.0
        s.car_pos[mask, 1] = pos
        yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = torch.rand(n, device=self.device)

    def _reset_aerial_challenge(self, mask, n):
        """Both on ground, ball high between them, high boost."""
        s = self.state

        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 2000 - 1000
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 1000 + 500
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 2] = torch.rand(n, device=self.device) * 200 - 100
        s.ball_vel[mask] = ball_vel

        for car_idx, y_lo, y_hi, yaw_val in [(0, -3000, -500, math.pi / 2),
                                               (1, 500, 3000, -math.pi / 2)]:
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = ball_pos[:, 0] + torch.rand(n, device=self.device) * 1000 - 500
            pos[:, 1] = torch.rand(n, device=self.device) * (y_hi - y_lo) + y_lo
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device),
                torch.full((n,), yaw_val, device=self.device),
                torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device) * 0.3 + 0.7

    def _reset_wall_play_1v1(self, mask, n):
        """One car on wall, opponent on ground, ball on wall."""
        s = self.state

        # Blue on right wall
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = ARENA_HALF_X - 50.0
        pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 2] = torch.rand(n, device=self.device) * 800 + 200
        s.car_pos[mask, 0] = pos
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device))
        s.car_on_ground[mask, 0] = 1.0
        s.car_surface_normal[mask, 0] = torch.tensor([-1.0, 0.0, 0.0], device=self.device)
        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.5 + 0.5

        # Ball on wall near blue
        ball_pos = pos.clone()
        ball_pos[:, 0] = ARENA_HALF_X - BALL_RADIUS - 10
        ball_pos[:, 2] += torch.rand(n, device=self.device) * 200
        s.ball_pos[mask] = ball_pos

        # Orange on ground
        pos_o = torch.zeros(n, 3, device=self.device)
        pos_o[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos_o[:, 1] = torch.rand(n, device=self.device) * 4000
        pos_o[:, 2] = 17.0
        s.car_pos[mask, 1] = pos_o
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = torch.rand(n, device=self.device)

    def _reset_shooting_practice(self, mask, n):
        """Blue has ball near opponent goal, orange in net."""
        s = self.state

        # Ball near orange goal
        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 2000 - 1000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 1500 + 3000  # 3000-4500
        ball_pos[:, 2] = BALL_RADIUS + 1.0
        s.ball_pos[mask] = ball_pos

        # Blue behind ball, facing goal
        pos = ball_pos.clone()
        pos[:, 1] -= torch.rand(n, device=self.device) * 500 + 200
        pos[:, 2] = 17.0
        s.car_pos[mask, 0] = pos
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.5 + 0.3

        # Orange in goal area
        pos_o = torch.zeros(n, 3, device=self.device)
        pos_o[:, 0] = torch.rand(n, device=self.device) * 1000 - 500
        pos_o[:, 1] = torch.rand(n, device=self.device) * 500 + 4500
        pos_o[:, 2] = 17.0
        s.car_pos[mask, 1] = pos_o
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = torch.rand(n, device=self.device) * 0.3

    def _reset_saving_practice(self, mask, n):
        """Orange shooting on blue goal, blue in goal area."""
        s = self.state

        # Ball near blue goal, moving toward goal
        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 1500 - 750
        ball_pos[:, 1] = -(torch.rand(n, device=self.device) * 1500 + 2500)  # -2500 to -4000
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 300 + BALL_RADIUS
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 1000 - 500
        ball_vel[:, 1] = -(torch.rand(n, device=self.device) * 1500 + 500)  # toward blue goal
        s.ball_vel[mask] = ball_vel

        # Blue in goal area, facing ball
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 800 - 400
        pos[:, 1] = -(torch.rand(n, device=self.device) * 500 + 4500)
        pos[:, 2] = 17.0
        s.car_pos[mask, 0] = pos
        to_ball = ball_pos[:, :2] - pos[:, :2]
        yaw = torch.atan2(to_ball[:, 1], to_ball[:, 0])
        s.car_quat[mask, 0] = quat_from_euler(
            torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.5 + 0.3

        # Orange behind ball, attacking
        pos_o = ball_pos.clone()
        pos_o[:, 1] += torch.rand(n, device=self.device) * 500 + 300
        pos_o[:, 2] = 17.0
        s.car_pos[mask, 1] = pos_o
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = torch.rand(n, device=self.device) * 0.5 + 0.3

    # ═══════════════════════════════════════════════════
    # Stage 3: 2v2 — mixed scenarios
    # ═══════════════════════════════════════════════════

    def _reset_stage3_scenarios(self, mask, n):
        """Stage 3 (2v2): kickoff + ground + aerial + ceiling mix."""
        r = torch.rand(n, device=self.device)
        indices = mask.nonzero(as_tuple=True)[0]

        # Kickoff 40%, Ground 30%, Aerial 20%, Ceiling 10%
        thresholds = [0.40, 0.70, 0.90, 1.00]

        for idx, (lo, hi, fn) in enumerate([
            (0.0, 0.40, self._reset_kickoff_2v2),
            (0.40, 0.70, self._reset_ground_2v2),
            (0.70, 0.90, self._reset_aerial_2v2),
            (0.90, 1.00, self._reset_ceiling_2v2),
        ]):
            local_mask = (r >= lo) & (r < hi)
            if not local_mask.any():
                continue
            full_mask = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            full_mask[indices[local_mask]] = True
            fn(full_mask, full_mask.sum().item())

    def _reset_kickoff_2v2(self, mask, n):
        """Standard 2v2 kickoff."""
        s = self.state
        for team_offset, yaw, y_sign in [(0, math.pi / 2, 1.0), (2, -math.pi / 2, -1.0)]:
            for car_local in range(2):
                car_idx = team_offset + car_local
                pos_idx = torch.randint(0, 5, (n,), device=self.device)
                positions = self._kickoff_pos[pos_idx]
                if y_sign < 0:
                    positions = positions * torch.tensor([-1.0, -1.0, 1.0], device=self.device)
                s.car_pos[mask, car_idx] = positions
                yaw_t = torch.full((n,), yaw, device=self.device)
                s.car_quat[mask, car_idx] = quat_from_euler(
                    torch.zeros(n, device=self.device), yaw_t, torch.zeros(n, device=self.device))

    def _reset_ground_2v2(self, mask, n):
        """Random ground play for 2v2."""
        s = self.state

        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 8000 - 4000
        ball_pos[:, 2] = BALL_RADIUS + 1.0
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 3000 - 1500
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 3000 - 1500
        s.ball_vel[mask] = ball_vel

        for car_idx in range(2):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 7000 - 3500
            pos[:, 1] = torch.rand(n, device=self.device) * 4500 - 4500
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos
            yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device)

        for car_idx in range(2, 4):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 7000 - 3500
            pos[:, 1] = torch.rand(n, device=self.device) * 4500
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos
            yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device)

    def _reset_aerial_2v2(self, mask, n):
        """Ball in air, cars on ground with high boost (2v2)."""
        s = self.state

        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 5000 - 2500
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 5000 - 2500
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 1200 + 400
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 800 - 400
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 800 - 400
        ball_vel[:, 2] = torch.rand(n, device=self.device) * 400 - 200
        s.ball_vel[mask] = ball_vel

        for car_idx in range(2):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
            pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 4000
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device),
                torch.full((n,), math.pi / 2, device=self.device),
                torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device) * 0.3 + 0.7

        for car_idx in range(2, 4):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
            pos[:, 1] = torch.rand(n, device=self.device) * 4000
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device),
                torch.full((n,), -math.pi / 2, device=self.device),
                torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device) * 0.3 + 0.7

    def _reset_ceiling_2v2(self, mask, n):
        """Ball high, one car per team near ceiling (2v2)."""
        s = self.state

        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 800 + 1000
        s.ball_pos[mask] = ball_pos

        # Blue car 0: near ceiling
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 1] = torch.rand(n, device=self.device) * 3000 - 3000
        pos[:, 2] = 1900.0
        s.car_pos[mask, 0] = pos
        s.car_quat[mask, 0] = quat_from_euler(
            torch.full((n,), math.pi, device=self.device),
            torch.full((n,), math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 0] = torch.rand(n, device=self.device) * 0.5 + 0.5
        s.car_on_ground[mask, 0] = 0.0

        # Blue car 1: on ground
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
        pos[:, 1] = torch.rand(n, device=self.device) * 2000 - 4000
        pos[:, 2] = 17.0
        s.car_pos[mask, 1] = pos
        s.car_quat[mask, 1] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 1] = torch.rand(n, device=self.device) * 0.5 + 0.3

        # Orange car 0: near ceiling
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        pos[:, 1] = torch.rand(n, device=self.device) * 3000
        pos[:, 2] = 1900.0
        s.car_pos[mask, 2] = pos
        s.car_quat[mask, 2] = quat_from_euler(
            torch.full((n,), math.pi, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 2] = torch.rand(n, device=self.device) * 0.5 + 0.5
        s.car_on_ground[mask, 2] = 0.0

        # Orange car 1: on ground
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
        pos[:, 1] = torch.rand(n, device=self.device) * 2000 + 2000
        pos[:, 2] = 17.0
        s.car_pos[mask, 3] = pos
        s.car_quat[mask, 3] = quat_from_euler(
            torch.zeros(n, device=self.device),
            torch.full((n,), -math.pi / 2, device=self.device),
            torch.zeros(n, device=self.device))
        s.car_boost[mask, 3] = torch.rand(n, device=self.device) * 0.5 + 0.3

    # ═══════════════════════════════════════════════════
    # Step / Physics / Terminal
    # ═══════════════════════════════════════════════════

    def step(self, actions):
        """Step all environments by tick_skip physics ticks.

        actions: (E, A, 8) float tensor
        Returns: (E,) bool tensor of terminal environments.
        """
        self.state.prev_ball_speed = self.state.ball_vel.norm(dim=-1)
        self.state.car_ball_touched[:] = 0.0

        for _ in range(self.tick_skip):
            self._physics_tick(actions)

        self.state.step_count += 1
        return self._check_terminals()

    def _physics_tick(self, actions):
        """Single 120Hz physics tick."""
        s = self.state
        apply_car_controls(s, actions, DT)
        integrate_positions(s, DT)
        arena_collide_ball(s)
        arena_collide_cars(s)
        touches = ball_car_collision(s)
        s.car_ball_touched = torch.max(s.car_ball_touched, touches)
        car_car_collision(s)
        self._update_boost_pads(DT)
        update_demoed_cars(s, DT)
        update_rotation_vectors(s)
        self._detect_goals()

    def _update_boost_pads(self, dt):
        """Update boost pad respawn timers and handle pickups."""
        s = self.state
        A = self.n_agents

        active_timers = s.boost_pad_timers > 0
        s.boost_pad_timers -= dt * active_timers.float()
        s.boost_pad_timers.clamp_(min=0)

        car_xy = s.car_pos[:, :, :2]  # (E, A, 2)
        alive = s.car_is_demoed < 0.5

        pad_xy = self._pad_pos_xy.unsqueeze(0).unsqueeze(0)  # (1, 1, 34, 2)
        diff = car_xy.unsqueeze(2) - pad_xy  # (E, A, 34, 2)
        dist = diff.norm(dim=-1)  # (E, A, 34)

        in_range = dist < BOOST_PAD_PICKUP_RADIUS
        available = s.boost_pad_timers.unsqueeze(1) <= 0
        can_pickup = in_range & available & alive.unsqueeze(2)

        pad_amount = self._pad_amount.unsqueeze(0).unsqueeze(0)
        boost_gain = can_pickup.float() * pad_amount
        total_boost_gain = boost_gain.sum(dim=2)
        s.car_boost += total_boost_gain
        s.car_boost.clamp_(max=1.0)

        any_pickup = can_pickup.any(dim=1)
        pad_respawn = self._pad_respawn.unsqueeze(0)
        s.boost_pad_timers = torch.where(
            any_pickup,
            pad_respawn.expand_as(s.boost_pad_timers),
            s.boost_pad_timers,
        )

    def _detect_goals(self):
        """Detect goals."""
        s = self.state
        ball_y = s.ball_pos[:, 1]
        ball_x_abs = torch.abs(s.ball_pos[:, 0])
        ball_z = s.ball_pos[:, 2]

        in_goal_opening = (ball_x_abs < GOAL_HALF_WIDTH) & (ball_z < GOAL_HEIGHT)

        orange_goal = (ball_y > ARENA_HALF_Y) & in_goal_opening
        if orange_goal.any():
            s.blue_score[orange_goal] += 1
            self._credit_goal(orange_goal, team=0)

        blue_goal = (ball_y < -ARENA_HALF_Y) & in_goal_opening
        if blue_goal.any():
            s.orange_score[blue_goal] += 1
            self._credit_goal(blue_goal, team=1)

    def _credit_goal(self, goal_mask, team):
        """Credit goal to scoring team."""
        s = self.state
        blue_cars = self.layout["blue_cars"]
        orange_cars = self.layout["orange_cars"]

        if team == 0 and blue_cars:
            s.match_goals[goal_mask, blue_cars[0]] += 1.0
        elif team == 1 and orange_cars:
            s.match_goals[goal_mask, orange_cars[0]] += 1.0

    def _check_terminals(self):
        """Check for episode termination."""
        s = self.state

        goal = (s.blue_score != self._prev_blue_score) | (s.orange_score != self._prev_orange_score)
        self._prev_blue_score[:] = s.blue_score
        self._prev_orange_score[:] = s.orange_score

        timeout = s.step_count >= self.timeout
        return goal | timeout

    def reset_done_envs(self, terminals):
        """Reset done environments."""
        if terminals.any():
            self._reset_envs(terminals)

    def set_stage(self, stage):
        """Update curriculum stage. Re-creates state if n_agents changes."""
        old_n_agents = self.n_agents
        self.stage = stage
        cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[0])
        self.tick_skip = cfg["tick_skip"]
        self.timeout = cfg["timeout"]
        new_n_agents = cfg.get("n_agents", 4)

        if new_n_agents != old_n_agents:
            self.n_agents = new_n_agents
            self.layout = get_agent_layout(new_n_agents)
            # n_envs might also change — handled by collector
