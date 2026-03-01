"""rewards.py — All reward signals + events + team spirit on GPU.

Supports variable n_agents (1v0, 1v1, 2v2).
Operates on TensorState directly (no data movement).
"""

import torch
import math
from .constants import (
    BALL_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED, BACK_NET_Y,
    GOAL_HEIGHT, BLUE_GOAL_BACK, ORANGE_GOAL_BACK,
)

# ── Reward weights per stage ──
# [R1:VelBallGoal, R2:BallGoalDist, R3:TouchQuality, R4:PlayerBallProxVel,
#  R5:Kickoff, R6:DefensivePos, R7:BoostEff, R8:DemoAttempt,
#  R9:AirControl, R10:FlipReset, R11:AngVel, R12:Speed, R13:BallAccel,
#  R14:SpeedFlip, R15:WaveDash, R16:WallDrive, R17:AirDribble]
STAGE_WEIGHTS = {
    0: [2.0, 1.0, 2.0, 2.0, 3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,   1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    1: [2.0, 1.0, 2.0, 2.0, 3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,   1.0, 0.5, 3.0, 2.0, 1.0, 0.0],
    2: [3.0, 1.5, 2.5, 1.5, 2.0, 0.0, 0.5, 0.0, 1.5, 0.0, 0.005, 1.0, 1.5, 2.0, 2.0, 1.0, 0.0],
    3: [5.0, 3.0, 3.0, 1.5, 2.0, 0.5, 1.0, 0.5, 1.5, 0.0, 0.005, 0.5, 2.0, 2.0, 2.0, 1.0, 0.0],
    4: [5.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.0, 1.0, 2.0, 5.0, 0.005, 0.3, 2.0, 1.0, 1.0, 0.5, 3.0],
    5: [5.0, 3.0, 3.0, 0.5, 2.0, 1.5, 1.0, 1.0, 2.0,10.0, 0.005, 0.2, 2.0, 0.5, 0.5, 0.5, 1.0],
}

EVENT_WEIGHTS = {
    0: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    1: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    2: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    3: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    4: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    5: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
}

TEAM_SPIRIT = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.6}


class GPURewards:
    """Computes reward signals for all envs/agents in batched PyTorch.

    Supports n_agents = 1, 2, or 4.
    """

    def __init__(self, n_envs, device='cuda', n_agents=4, layout=None):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.device = device
        self.layout = layout

        # Event tracking: [goals, team_score, opp_score, touched, shots, saves, demos, boost]
        self.event_last = torch.zeros(n_envs, n_agents, 8, device=device)

        # Kickoff state
        self.is_kickoff = torch.zeros(n_envs, dtype=torch.bool, device=device)

        # Previous boost for R7
        self.prev_player_boost = torch.zeros(n_envs, n_agents, device=device)

        # Tracking state for new rewards
        self.prev_on_ground = torch.ones(n_envs, n_agents, device=device)
        self.prev_has_flipped = torch.zeros(n_envs, n_agents, device=device)

        # Pre-compute goal positions for broadcasting
        self._orange_goal = ORANGE_GOAL_BACK.to(device)
        self._blue_goal = BLUE_GOAL_BACK.to(device)

        # Pre-allocate weight tensors
        self._stage_weights = {}
        self._event_weights = {}
        for s in range(6):
            self._stage_weights[s] = torch.tensor(STAGE_WEIGHTS[s], device=device)
            self._event_weights[s] = torch.tensor(EVENT_WEIGHTS[s], device=device)

    def reset_envs(self, mask, state):
        """Reset reward state for terminated envs. mask: (E,) bool."""
        if not mask.any():
            return

        A = self.n_agents
        is_blue = state.car_team == 0  # (E, A)

        goals = state.match_goals[mask]
        team_score = torch.where(is_blue[mask], state.blue_score[mask].unsqueeze(1).float(),
                                 state.orange_score[mask].unsqueeze(1).float())
        opp_score = torch.where(is_blue[mask], state.orange_score[mask].unsqueeze(1).float(),
                                state.blue_score[mask].unsqueeze(1).float())
        touched = state.car_ball_touched[mask]
        shots = state.match_shots[mask]
        saves = state.match_saves[mask]
        demos = state.match_demos[mask]
        boost = state.car_boost[mask]

        self.event_last[mask, :, 0] = goals
        self.event_last[mask, :, 1] = team_score
        self.event_last[mask, :, 2] = opp_score
        self.event_last[mask, :, 3] = touched
        self.event_last[mask, :, 4] = shots
        self.event_last[mask, :, 5] = saves
        self.event_last[mask, :, 6] = demos
        self.event_last[mask, :, 7] = boost

        # Kickoff detection
        bp = state.ball_pos[mask]
        self.is_kickoff[mask] = (
            (torch.abs(bp[:, 0]) < 50) &
            (torch.abs(bp[:, 1]) < 50) &
            (bp[:, 2] < 120)
        )

        self.prev_player_boost[mask] = state.car_boost[mask]
        self.prev_on_ground[mask] = state.car_on_ground[mask]
        self.prev_has_flipped[mask] = state.car_has_flipped[mask]

    def compute(self, state, stage):
        """Compute combined rewards for all envs/agents.

        Returns: (E, A) reward tensor on GPU.
        """
        E = self.n_envs
        A = self.n_agents
        s = state
        layout = self.layout
        weights = self._stage_weights.get(stage, self._stage_weights[5])
        ev_weights = self._event_weights.get(stage, self._event_weights[5])
        ts = TEAM_SPIRIT.get(stage, 0.6)

        # ── Precompute common values ──
        ball_e = s.ball_pos.unsqueeze(1)  # (E, 1, 3)
        to_ball = ball_e - s.car_pos      # (E, A, 3)
        to_ball_dist = to_ball.norm(dim=-1)  # (E, A)
        to_ball_dir = to_ball / (to_ball_dist.unsqueeze(-1) + 1e-6)

        player_speed = s.car_vel.norm(dim=-1)  # (E, A)
        is_blue = (s.car_team == 0)
        is_blue_f = is_blue.float()

        # Goal positions per agent
        opp_goal = (is_blue_f.unsqueeze(-1) * self._orange_goal +
                    (1 - is_blue_f).unsqueeze(-1) * self._blue_goal)
        own_goal = (is_blue_f.unsqueeze(-1) * self._blue_goal +
                    (1 - is_blue_f).unsqueeze(-1) * self._orange_goal)

        # Closest on team — for 1v0, always closest; for 1v1/2v2, compare with ally
        if A == 1:
            is_closest = torch.ones(E, A, dtype=torch.bool, device=self.device)
        elif A == 2:
            # No ally in 1v1, always closest on team
            is_closest = torch.ones(E, A, dtype=torch.bool, device=self.device)
        else:
            ally_idx = layout["ally_idx"]
            ally_dist = torch.stack([to_ball_dist[:, ally_idx[i]] for i in range(A)], dim=1)
            is_closest = to_ball_dist <= ally_dist

        ball_speed = s.ball_vel.norm(dim=-1)  # (E,)

        rewards = torch.zeros(E, A, device=self.device)

        # ── R1: VelocityBallToGoal ──
        if weights[0] > 0:
            pos_diff = opp_goal - ball_e
            norm_pd = pos_diff / (pos_diff.norm(dim=-1, keepdim=True) + 1e-6)
            norm_bv = s.ball_vel.unsqueeze(1) / BALL_MAX_SPEED
            rewards += weights[0] * (norm_pd * norm_bv).sum(dim=-1)

        # ── R2: BallGoalDistancePotential ──
        if weights[1] > 0:
            ball_to_opp = (ball_e - opp_goal).norm(dim=-1)
            ball_to_own = (ball_e - own_goal).norm(dim=-1)
            rewards += weights[1] * (torch.exp(-ball_to_opp / 6000.0) - torch.exp(-ball_to_own / 6000.0))

        # ── R3: TouchQuality ──
        if weights[2] > 0:
            touched = s.car_ball_touched
            ball_z = s.ball_pos[:, 2].unsqueeze(1)
            ball_speed_e = ball_speed.unsqueeze(1)
            height_term = 1.0 + torch.clamp(ball_z - 150.0, min=0).pow(1.0/3.0) / (2044.0 ** (1.0/3.0)) * 2.0
            speed_term = 0.5 + 0.5 * (ball_speed_e / 2300.0).clamp(max=2.0)
            wall_x = (torch.abs(s.ball_pos[:, 0]) > 3800.0).unsqueeze(1)
            wall_y = (torch.abs(s.ball_pos[:, 1]) > 4800.0).unsqueeze(1)
            wall_factor = torch.where(wall_x | wall_y, torch.tensor(1.5, device=self.device),
                                      torch.tensor(1.0, device=self.device))
            rewards += weights[2] * (touched * height_term * speed_term * wall_factor)

        # ── R4: PlayerBallProximityVelocity ──
        if weights[3] > 0:
            speed_toward_ball = (s.car_vel * to_ball_dir).sum(dim=-1) / CAR_MAX_SPEED
            speed_toward_ball = speed_toward_ball.clamp(min=0)
            rewards += weights[3] * (speed_toward_ball * is_closest.float())

        # ── R5: KickoffReward ──
        if weights[4] > 0:
            self.is_kickoff = self.is_kickoff & (ball_speed < 100)
            kick_speed = (s.car_vel * to_ball_dir).sum(dim=-1)
            kick_r = (kick_speed / 2300.0).clamp(min=0) + torch.exp(-to_ball_dist / 800.0)
            rewards += weights[4] * (kick_r * self.is_kickoff.unsqueeze(1).float())
        else:
            self.is_kickoff = self.is_kickoff & (ball_speed < 100)

        # ── R6: DefensivePositioning (skip for 1v0, reduced for 1v1) ──
        if weights[5] > 0 and A >= 2:
            is_support = ~is_closest
            own_goal_y = torch.where(is_blue, torch.tensor(-5120.0, device=self.device),
                                     torch.tensor(5120.0, device=self.device))
            own_goal_3d = torch.zeros(E, A, 3, device=self.device)
            own_goal_3d[:, :, 1] = own_goal_y

            g2b = ball_e - own_goal_3d
            g2p = s.car_pos - own_goal_3d
            g2b_n = g2b.norm(dim=-1, keepdim=True) + 1e-6
            g2p_n = g2p.norm(dim=-1, keepdim=True) + 1e-6

            align = ((g2p * g2b).sum(dim=-1) / (g2b_n.squeeze(-1) * g2p_n.squeeze(-1))).clamp(min=0)
            dist_ratio = g2p_n.squeeze(-1) / g2b_n.squeeze(-1)
            gaussian = torch.exp(-((dist_ratio - 0.7) ** 2) / (2.0 * 0.15 ** 2))
            rewards += weights[5] * (is_support.float() * align * gaussian)

        # ── R7: BoostEfficiency ──
        if weights[6] > 0:
            boost_gained = (s.car_boost - self.prev_player_boost).clamp(min=0)
            is_small = (boost_gained > 0.01) & (boost_gained <= 0.15)
            pad_mult = torch.where(is_small, torch.tensor(2.0, device=self.device),
                                   torch.tensor(1.0, device=self.device))
            rewards += weights[6] * (torch.sqrt(boost_gained) * pad_mult).clamp(max=0.5)

        # ── R8: DemoAttempt (skip for 1v0) ──
        if weights[7] > 0 and A >= 2:
            enemy0_idx = layout["enemy0_idx"]
            for i in range(A):
                e0 = enemy0_idx[i]
                if e0 < 0:
                    continue
                to_opp = s.car_pos[:, e0] - s.car_pos[:, i]
                d = to_opp.norm(dim=-1)
                opp_dir = to_opp / (d.unsqueeze(-1) + 1e-6)
                speed_to_opp = (s.car_vel[:, i] * opp_dir).sum(dim=-1)
                rewards[:, i] += weights[7] * (
                    torch.exp(-d / 500.0) *
                    (speed_to_opp / 2300.0).clamp(min=0) *
                    (player_speed[:, i] > 1500.0).float()
                )

        # ── R9: AirControl ──
        if weights[8] > 0:
            bz = s.ball_pos[:, 2].unsqueeze(1)
            cz = s.car_pos[:, :, 2]
            ball_above = bz > (cz + 60.0)

            xy_diff = s.ball_pos[:, :2].unsqueeze(1) - s.car_pos[:, :, :2]
            xy_dist = xy_diff.norm(dim=-1)
            close_overhead = xy_dist < 180.0
            prox = (1.0 - xy_dist / 180.0).clamp(0, 1)
            ht = ((bz - cz - 60.0) / 250.0).clamp(0, 1)

            dribble = torch.where(
                ball_above & close_overhead & (s.car_on_ground > 0.5),
                prox * (0.3 + 0.7 * ht),
                torch.zeros_like(prox),
            )

            off_ground = (s.car_on_ground < 0.5).float()
            facing_ball = (s.car_fwd * to_ball_dir).sum(dim=-1).clamp(min=0)
            aerial = off_ground * facing_ball
            rewards += weights[8] * torch.max(dribble, aerial)

        # ── R10: FlipResetDetector ──
        if weights[9] > 0:
            car_up_z = s.car_up[:, :, 2]
            rewards += weights[9] * (
                s.car_ball_touched *
                (s.car_on_ground < 0.5).float() *
                (car_up_z < -0.5).float() * 10.0
            )

        # ── R11: AngularVelocity ──
        if weights[10] > 0:
            rewards += weights[10] * (s.car_ang_vel.norm(dim=-1) / (6.0 * math.pi))

        # ── R12: Speed + Anti-Passive ──
        if weights[11] > 0:
            speed_ratio = player_speed / CAR_MAX_SPEED
            passive_pen = (1.0 - player_speed / 600.0).clamp(min=0) * 0.03
            passive_pen *= (s.car_is_demoed < 0.5).float() * (s.car_on_ground > 0.5).float()
            rewards += weights[11] * (speed_ratio - passive_pen)

        # ── R13: BallAcceleration ──
        if weights[12] > 0:
            ball_accel = (ball_speed - s.prev_ball_speed).clamp(min=0)
            accel_norm = ball_accel / BALL_MAX_SPEED
            rewards += weights[12] * (accel_norm.unsqueeze(1) * s.car_ball_touched)

        # ── R14: SpeedFlipReward ──
        # Detect: has_flipped & high forward speed & near ground & in air
        if weights[13] > 0:
            has_flipped = s.car_has_flipped > 0.5
            fwd_speed = (s.car_vel * s.car_fwd).sum(dim=-1)  # (E, A)
            near_ground = s.car_pos[:, :, 2] < 100.0
            in_air = s.car_on_ground < 0.5
            speed_bonus = (fwd_speed / CAR_MAX_SPEED).clamp(min=0)
            rewards += weights[13] * (
                has_flipped.float() * in_air.float() * near_ground.float() *
                (fwd_speed > 1000.0).float() * speed_bonus
            )

        # ── R15: WaveDashReward ──
        # Detect: just landed & was flipping last frame & high speed
        if weights[14] > 0:
            just_landed = (s.car_on_ground > 0.5) & (self.prev_on_ground < 0.5)
            was_flipping = self.prev_has_flipped > 0.5
            current_speed = s.car_vel.norm(dim=-1)
            speed_bonus = (current_speed / CAR_MAX_SPEED).clamp(min=0)
            rewards += weights[14] * (
                just_landed.float() * was_flipping.float() *
                (current_speed > 800.0).float() * speed_bonus
            )

        # ── R16: WallDriveReward ──
        # Detect: on ground & surface normal is not floor (abs(z) < 0.5)
        if weights[15] > 0:
            on_wall = (s.car_on_ground > 0.5) & (torch.abs(s.car_surface_normal[:, :, 2]) < 0.5)
            fwd_speed = (s.car_vel * s.car_fwd).sum(dim=-1)
            wall_speed = (torch.abs(fwd_speed) / CAR_MAX_SPEED).clamp(min=0)
            rewards += weights[15] * (on_wall.float() * wall_speed)

        # ── R17: AirDribbleReward ──
        # Detect: in air & close to ball & ball high & touched
        if weights[16] > 0:
            in_air = s.car_on_ground < 0.5
            ball_high = s.ball_pos[:, 2].unsqueeze(1) > 300.0  # (E, 1)
            close = to_ball_dist < 300.0  # (E, A)
            touched = s.car_ball_touched > 0.5
            rewards += weights[16] * (
                in_air.float() * ball_high.float() * close.float() * touched.float() * 2.0
            )

        # ── Event Reward ──
        current_ev = torch.zeros(E, A, 8, device=self.device)
        current_ev[:, :, 0] = s.match_goals
        for j in range(A):
            blue_mask = (s.car_team[:, j] == 0).float()
            current_ev[:, j, 1] = blue_mask * s.blue_score.float() + (1 - blue_mask) * s.orange_score.float()
            current_ev[:, j, 2] = blue_mask * s.orange_score.float() + (1 - blue_mask) * s.blue_score.float()
        current_ev[:, :, 3] = s.car_ball_touched
        current_ev[:, :, 4] = s.match_shots
        current_ev[:, :, 5] = s.match_saves
        current_ev[:, :, 6] = s.match_demos
        current_ev[:, :, 7] = s.car_boost

        diff = (current_ev - self.event_last).clamp(min=0)
        rewards += (diff * ev_weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        self.event_last.copy_(current_ev)

        # ── Team Spirit blending (only for 2v2) ──
        if ts > 0 and A == 4:
            blue_mean = (rewards[:, 0] + rewards[:, 1]) * 0.5
            orange_mean = (rewards[:, 2] + rewards[:, 3]) * 0.5
            team_mean = torch.stack([blue_mean, blue_mean, orange_mean, orange_mean], dim=1)
            rewards = (1.0 - ts) * rewards + ts * team_mean

        # ── Update tracking state ──
        self.prev_player_boost.copy_(s.car_boost)
        self.prev_on_ground.copy_(s.car_on_ground)
        self.prev_has_flipped.copy_(s.car_has_flipped)

        return rewards
