"""environment.py — GPUEnvironment: step, reset, terminal detection.

Orchestrates physics ticks, collision detection, boost pad pickup,
and episode management. All operations are batched on GPU.
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
    KICKOFF_POSITIONS, STAGE_CONFIG,
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

    All state lives on GPU as tensors. A single step() call advances all
    E environments through tick_skip physics sub-steps.
    """

    def __init__(self, n_envs, device='cuda', stage=0):
        self.n_envs = n_envs
        self.device = device
        self.stage = stage
        self.state = TensorState(n_envs, device)

        cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[0])
        self.tick_skip = cfg["tick_skip"]
        self.timeout = cfg["timeout"]

        # Pre-move boost pad positions to device
        self._pad_pos = BOOST_PAD_POSITIONS.to(device)  # (34, 3)
        self._pad_pos_xy = self._pad_pos[:, :2]         # (34, 2) for distance check

        # Boost pad respawn times (large=10s, small=4s)
        self._pad_respawn = torch.zeros(N_BOOST_PADS, device=device)
        self._pad_respawn[:N_LARGE_PADS] = LARGE_PAD_RESPAWN
        self._pad_respawn[N_LARGE_PADS:] = SMALL_PAD_RESPAWN

        # Boost pad amounts (large=1.0, small=0.12)
        self._pad_amount = torch.zeros(N_BOOST_PADS, device=device)
        self._pad_amount[:N_LARGE_PADS] = LARGE_PAD_BOOST
        self._pad_amount[N_LARGE_PADS:] = SMALL_PAD_BOOST

        # Kickoff positions on device
        self._kickoff_pos = KICKOFF_POSITIONS.to(device)

        # Goal tracking for terminal detection
        self._prev_blue_score = torch.zeros(n_envs, dtype=torch.long, device=device)
        self._prev_orange_score = torch.zeros(n_envs, dtype=torch.long, device=device)

        # Initialize all environments
        self.reset_all()

    def reset_all(self):
        """Reset all environments to initial state."""
        mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        self._reset_envs(mask)
        self._prev_blue_score[:] = 0
        self._prev_orange_score[:] = 0

    def _reset_envs(self, mask):
        """Reset specific environments. mask: (E,) bool tensor."""
        if not mask.any():
            return

        n_reset = mask.sum().item()
        s = self.state

        # ── Ball: center of field at resting height ──
        s.ball_pos[mask] = torch.tensor([0.0, 0.0, BALL_RADIUS + 1.0], device=self.device)
        s.ball_vel[mask] = 0.0
        s.ball_ang_vel[mask] = 0.0

        # ── Scenario selection ──
        # For now: pure kickoff (stage 0) or mixed (later stages)
        if self.stage == 0:
            self._reset_kickoff(mask, n_reset)
        else:
            # Mixed scenarios
            r = torch.rand(n_reset, device=self.device)
            kickoff_mask_local = r < 0.4
            ground_mask_local = (r >= 0.4) & (r < 0.7)
            aerial_mask_local = (r >= 0.7) & (r < 0.9)
            ceiling_mask_local = r >= 0.9

            # Create full-size masks from local (n_reset) masks
            indices = mask.nonzero(as_tuple=True)[0]

            kickoff_full = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            ground_full = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            aerial_full = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
            ceiling_full = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)

            kickoff_full[indices[kickoff_mask_local]] = True
            ground_full[indices[ground_mask_local]] = True
            aerial_full[indices[aerial_mask_local]] = True
            ceiling_full[indices[ceiling_mask_local]] = True

            if kickoff_full.any():
                self._reset_kickoff(kickoff_full, kickoff_full.sum().item())
            if ground_full.any():
                self._reset_ground(ground_full, ground_full.sum().item())
            if aerial_full.any():
                self._reset_aerial(aerial_full, aerial_full.sum().item())
            if ceiling_full.any():
                self._reset_ceiling(ceiling_full, ceiling_full.sum().item())

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

        # Reset ball speed tracking
        s.prev_ball_speed[mask] = 0.0

        # Score tracking
        self._prev_blue_score[mask] = 0
        self._prev_orange_score[mask] = 0

        # Update rotation vectors
        fwd, up = quat_to_fwd_up(s.car_quat)
        s.car_fwd = fwd
        s.car_up = up

    def _reset_kickoff(self, mask, n):
        """Standard kickoff: ball at center, cars at kickoff positions."""
        s = self.state
        # Ball already at center from parent

        # Random kickoff positions for each car
        # Blue team: 2 cars from 5 positions (facing +Y)
        # Orange team: 2 cars mirrored (facing -Y)
        for team_offset, yaw, y_sign in [(0, math.pi / 2, 1.0), (2, -math.pi / 2, -1.0)]:
            for car_local in range(2):
                car_idx = team_offset + car_local
                # Random position index per env
                pos_idx = torch.randint(0, 5, (n,), device=self.device)
                positions = self._kickoff_pos[pos_idx]  # (n, 3)
                if y_sign < 0:
                    positions = positions * torch.tensor([-1.0, -1.0, 1.0], device=self.device)
                s.car_pos[mask, car_idx] = positions

                # Set orientation
                yaw_t = torch.full((n,), yaw, device=self.device)
                pitch_t = torch.zeros(n, device=self.device)
                roll_t = torch.zeros(n, device=self.device)
                s.car_quat[mask, car_idx] = quat_from_euler(pitch_t, yaw_t, roll_t)

    def _reset_ground(self, mask, n):
        """Random ground play scenario."""
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

        # Blue cars in blue half
        for car_idx in range(2):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 7000 - 3500
            pos[:, 1] = torch.rand(n, device=self.device) * 4500 - 4500  # -4500 to 0
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos

            yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device)

        # Orange cars in orange half
        for car_idx in range(2, 4):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 7000 - 3500
            pos[:, 1] = torch.rand(n, device=self.device) * 4500  # 0 to 4500
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos

            yaw = torch.rand(n, device=self.device) * 2 * math.pi - math.pi
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device)

    def _reset_aerial(self, mask, n):
        """Ball in air, cars on ground with high boost."""
        s = self.state

        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 5000 - 2500
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 5000 - 2500
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 1200 + 400  # 400-1600
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 800 - 400
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 800 - 400
        ball_vel[:, 2] = torch.rand(n, device=self.device) * 400 - 200
        s.ball_vel[mask] = ball_vel

        # Blue cars
        for car_idx in range(2):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
            pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 4000
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos
            yaw = torch.full((n,), math.pi / 2, device=self.device)
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device) * 0.3 + 0.7

        # Orange cars
        for car_idx in range(2, 4):
            pos = torch.zeros(n, 3, device=self.device)
            pos[:, 0] = torch.rand(n, device=self.device) * 6000 - 3000
            pos[:, 1] = torch.rand(n, device=self.device) * 4000
            pos[:, 2] = 17.0
            s.car_pos[mask, car_idx] = pos
            yaw = torch.full((n,), -math.pi / 2, device=self.device)
            s.car_quat[mask, car_idx] = quat_from_euler(
                torch.zeros(n, device=self.device), yaw, torch.zeros(n, device=self.device))
            s.car_boost[mask, car_idx] = torch.rand(n, device=self.device) * 0.3 + 0.7

    def _reset_ceiling(self, mask, n):
        """Ball high, one car per team near ceiling."""
        s = self.state

        ball_pos = torch.zeros(n, 3, device=self.device)
        ball_pos[:, 0] = torch.rand(n, device=self.device) * 4000 - 2000
        ball_pos[:, 1] = torch.rand(n, device=self.device) * 4000 - 2000
        ball_pos[:, 2] = torch.rand(n, device=self.device) * 800 + 1000  # 1000-1800
        s.ball_pos[mask] = ball_pos

        ball_vel = torch.zeros(n, 3, device=self.device)
        ball_vel[:, 0] = torch.rand(n, device=self.device) * 600 - 300
        ball_vel[:, 1] = torch.rand(n, device=self.device) * 600 - 300
        ball_vel[:, 2] = torch.rand(n, device=self.device) * 300 - 100
        s.ball_vel[mask] = ball_vel

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

    def step(self, actions):
        """Step all environments by tick_skip physics ticks.

        actions: (E, 4, 8) float tensor
            [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]

        Returns: (E,) bool tensor of terminal environments.
        """
        # Save prev ball speed for R13 reward
        self.state.prev_ball_speed = self.state.ball_vel.norm(dim=-1)

        # Clear per-step touch flags
        self.state.car_ball_touched[:] = 0.0

        for _ in range(self.tick_skip):
            self._physics_tick(actions)

        # Increment step counter
        self.state.step_count += 1

        # Check terminals
        terminals = self._check_terminals()

        return terminals

    def _physics_tick(self, actions):
        """Single 120Hz physics tick for all environments."""
        s = self.state
        dt = DT

        # 1. Apply car controls (throttle, steer, boost, jump/flip)
        apply_car_controls(s, actions, dt)

        # 2. Integrate positions
        integrate_positions(s, dt)

        # 3. Arena collision (ball + cars vs walls/floor/ceiling)
        arena_collide_ball(s)
        arena_collide_cars(s)

        # 4. Ball-car collision
        touches = ball_car_collision(s)
        # Accumulate touches across sub-ticks (max so it's 0 or 1)
        s.car_ball_touched = torch.max(s.car_ball_touched, touches)

        # 5. Car-car collision (demos + bumps)
        car_car_collision(s)

        # 6. Boost pad pickup
        self._update_boost_pads(dt)

        # 7. Respawn demoed cars
        update_demoed_cars(s, dt)

        # 8. Update rotation vectors from quaternions
        update_rotation_vectors(s)

        # 9. Detect goals (ball crossing goal line)
        self._detect_goals()

    def _update_boost_pads(self, dt):
        """Update boost pad respawn timers and handle pickups."""
        s = self.state

        # Countdown respawn timers
        active_timers = s.boost_pad_timers > 0
        s.boost_pad_timers -= dt * active_timers.float()
        s.boost_pad_timers.clamp_(min=0)

        # Check pickup for each car
        # Car XY positions: (E, 4, 2)
        car_xy = s.car_pos[:, :, :2]  # (E, 4, 2)
        alive = s.car_is_demoed < 0.5  # (E, 4)

        # Pad XY positions: (34, 2) → broadcast to (1, 1, 34, 2)
        pad_xy = self._pad_pos_xy.unsqueeze(0).unsqueeze(0)  # (1, 1, 34, 2)

        # Distance: (E, 4, 34)
        diff = car_xy.unsqueeze(2) - pad_xy  # (E, 4, 34, 2)
        dist = diff.norm(dim=-1)  # (E, 4, 34)

        # In range and pad is available
        in_range = dist < BOOST_PAD_PICKUP_RADIUS  # (E, 4, 34)
        available = s.boost_pad_timers.unsqueeze(1) <= 0  # (E, 1, 34)
        can_pickup = in_range & available & alive.unsqueeze(2)  # (E, 4, 34)

        # For each pad, check if any car picks it up (first car wins)
        # Apply boost to cars that pick up
        pad_amount = self._pad_amount.unsqueeze(0).unsqueeze(0)  # (1, 1, 34)

        # Boost gained per car per pad
        boost_gain = can_pickup.float() * pad_amount  # (E, 4, 34)

        # Sum boost across all pads picked up by each car
        total_boost_gain = boost_gain.sum(dim=2)  # (E, 4)
        s.car_boost += total_boost_gain
        s.car_boost.clamp_(max=1.0)

        # Mark pads as used: any car picked up → set timer
        any_pickup = can_pickup.any(dim=1)  # (E, 34)
        pad_respawn = self._pad_respawn.unsqueeze(0)  # (1, 34)
        s.boost_pad_timers = torch.where(
            any_pickup,
            pad_respawn.expand_as(s.boost_pad_timers),
            s.boost_pad_timers,
        )

    def _detect_goals(self):
        """Detect goals: ball center past goal line inside goal opening."""
        s = self.state
        ball_y = s.ball_pos[:, 1]
        ball_x_abs = torch.abs(s.ball_pos[:, 0])
        ball_z = s.ball_pos[:, 2]

        in_goal_opening = (ball_x_abs < GOAL_HALF_WIDTH) & (ball_z < GOAL_HEIGHT)

        # Orange goal: blue scores (ball at +Y past goal line)
        orange_goal = (ball_y > ARENA_HALF_Y) & in_goal_opening
        if orange_goal.any():
            s.blue_score[orange_goal] += 1
            self._credit_goal(orange_goal, team=0)

        # Blue goal: orange scores (ball at -Y past goal line)
        blue_goal = (ball_y < -ARENA_HALF_Y) & in_goal_opening
        if blue_goal.any():
            s.orange_score[blue_goal] += 1
            self._credit_goal(blue_goal, team=1)

    def _credit_goal(self, goal_mask, team):
        """Credit goal to scoring team. Simplified: credit to car 0 of team."""
        s = self.state
        if team == 0:  # blue scored
            s.match_goals[goal_mask, 0] += 1.0  # blue car 0 gets credit
        else:  # orange scored
            s.match_goals[goal_mask, 2] += 1.0  # orange car 0 gets credit

    def _check_terminals(self):
        """Check for episode termination: goal scored or timeout.

        Returns: (E,) bool tensor.
        """
        s = self.state

        # Goal scored
        goal = (s.blue_score != self._prev_blue_score) | (s.orange_score != self._prev_orange_score)
        self._prev_blue_score[:] = s.blue_score
        self._prev_orange_score[:] = s.orange_score

        # Timeout
        timeout = s.step_count >= self.timeout

        return goal | timeout

    def reset_done_envs(self, terminals):
        """Reset environments that are done. Call after step().

        terminals: (E,) bool tensor from step().
        """
        if terminals.any():
            self._reset_envs(terminals)

    def set_stage(self, stage):
        """Update curriculum stage (tick_skip, timeout, scenario mix)."""
        self.stage = stage
        cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[0])
        self.tick_skip = cfg["tick_skip"]
        self.timeout = cfg["timeout"]
