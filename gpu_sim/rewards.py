"""rewards.py — All 13 reward signals + events + team spirit on GPU.

Direct port of vectorized_env.py VectorizedRewards from numpy to PyTorch.
Operates on TensorState directly (no data movement).
"""

import torch
import math
from .constants import (
    BALL_RADIUS, BALL_MAX_SPEED, CAR_MAX_SPEED, BACK_NET_Y,
    GOAL_HEIGHT, BLUE_GOAL_BACK, ORANGE_GOAL_BACK,
    ALLY_IDX, ENEMY0_IDX, ENEMY1_IDX,
)

# ── Reward weights per stage (same as vectorized_env.py) ──
# [R1:VelBallGoal, R2:BallGoalDist, R3:TouchQuality, R4:PlayerBallProxVel,
#  R5:Kickoff, R6:DefensivePos, R7:BoostEff, R8:DemoAttempt,
#  R9:AirControl, R10:FlipReset, R11:AngVel, R12:Speed, R13:BallAccel]
STAGE_WEIGHTS = {
    0: [2.0, 1.0, 2.0, 2.0, 3.0, 0.5, 0.5, 0.0, 0.5, 0.0,  0.0,   1.0, 1.5],
    1: [5.0, 3.0, 3.0, 1.5, 2.0, 1.5, 1.0, 0.5, 1.5, 0.0,  0.005, 0.5, 2.0],
    2: [5.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.0, 1.0, 2.0, 5.0,  0.005, 0.3, 2.0],
    3: [5.0, 3.0, 3.0, 0.5, 2.0, 1.5, 1.0, 1.0, 2.0, 10.0, 0.005, 0.2, 2.0],
}

EVENT_WEIGHTS = {
    0: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    1: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    2: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
    3: [10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0],
}

TEAM_SPIRIT = {0: 0.0, 1: 0.3, 2: 0.5, 3: 0.6}


class GPURewards:
    """Computes all 13 reward signals for all envs/agents in batched PyTorch."""

    def __init__(self, n_envs, device='cuda'):
        self.n_envs = n_envs
        self.n_agents = 4
        self.device = device

        # Event tracking: [goals, team_score, opp_score, touched, shots, saves, demos, boost]
        self.event_last = torch.zeros(n_envs, 4, 8, device=device)

        # Kickoff state
        self.is_kickoff = torch.zeros(n_envs, dtype=torch.bool, device=device)

        # Previous boost for R7
        self.prev_player_boost = torch.zeros(n_envs, 4, device=device)

        # Pre-compute goal positions for broadcasting
        self._orange_goal = ORANGE_GOAL_BACK.to(device)  # (3,)
        self._blue_goal = BLUE_GOAL_BACK.to(device)      # (3,)

        # Pre-allocate weight tensors
        self._stage_weights = {}
        self._event_weights = {}
        for s in range(4):
            self._stage_weights[s] = torch.tensor(STAGE_WEIGHTS[s], device=device)
            self._event_weights[s] = torch.tensor(EVENT_WEIGHTS[s], device=device)

    def reset_envs(self, mask, state):
        """Reset reward state for terminated envs. mask: (E,) bool."""
        if not mask.any():
            return

        # Snapshot event state
        is_blue = state.car_team == 0  # (E, 4)

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

    def compute(self, state, stage):
        """Compute combined rewards for all envs/agents.

        Returns: (E, 4) reward tensor on GPU.
        """
        E = self.n_envs
        s = state
        weights = self._stage_weights.get(stage, self._stage_weights[3])
        ev_weights = self._event_weights.get(stage, self._event_weights[3])
        ts = TEAM_SPIRIT.get(stage, 0.6)

        # ── Precompute common values ──
        ball_e = s.ball_pos.unsqueeze(1)  # (E, 1, 3)
        to_ball = ball_e - s.car_pos      # (E, 4, 3)
        to_ball_dist = to_ball.norm(dim=-1)  # (E, 4)
        to_ball_dir = to_ball / (to_ball_dist.unsqueeze(-1) + 1e-6)

        player_speed = s.car_vel.norm(dim=-1)  # (E, 4)
        is_blue = (s.car_team == 0)  # (E, 4) bool
        is_blue_f = is_blue.float()

        # Goal positions per agent
        opp_goal = (is_blue_f.unsqueeze(-1) * self._orange_goal +
                    (1 - is_blue_f).unsqueeze(-1) * self._blue_goal)  # (E, 4, 3)
        own_goal = (is_blue_f.unsqueeze(-1) * self._blue_goal +
                    (1 - is_blue_f).unsqueeze(-1) * self._orange_goal)

        # Closest on team
        ally_dist = to_ball_dist[:, ALLY_IDX]  # (E, 4)
        is_closest = to_ball_dist <= ally_dist  # (E, 4)

        # Ball speed
        ball_speed = s.ball_vel.norm(dim=-1)  # (E,)

        rewards = torch.zeros(E, 4, device=self.device)

        # ── R1: VelocityBallToGoal ──
        if weights[0] > 0:
            pos_diff = opp_goal - ball_e  # (E, 4, 3)
            norm_pd = pos_diff / (pos_diff.norm(dim=-1, keepdim=True) + 1e-6)
            norm_bv = s.ball_vel.unsqueeze(1) / BALL_MAX_SPEED  # (E, 1, 3)
            rewards += weights[0] * (norm_pd * norm_bv).sum(dim=-1)

        # ── R2: BallGoalDistancePotential ──
        if weights[1] > 0:
            ball_to_opp = (ball_e - opp_goal).norm(dim=-1)  # (E, 4)
            ball_to_own = (ball_e - own_goal).norm(dim=-1)
            rewards += weights[1] * (torch.exp(-ball_to_opp / 6000.0) - torch.exp(-ball_to_own / 6000.0))

        # ── R3: TouchQuality ──
        if weights[2] > 0:
            touched = s.car_ball_touched  # (E, 4)
            ball_z = s.ball_pos[:, 2].unsqueeze(1)  # (E, 1)
            ball_speed_e = ball_speed.unsqueeze(1)   # (E, 1)
            height_term = 1.0 + torch.clamp(ball_z - 150.0, min=0).pow(1.0/3.0) / (2044.0 ** (1.0/3.0)) * 2.0
            speed_term = 0.5 + 0.5 * (ball_speed_e / 2300.0).clamp(max=2.0)
            wall_x = (torch.abs(s.ball_pos[:, 0]) > 3800.0).unsqueeze(1)
            wall_y = (torch.abs(s.ball_pos[:, 1]) > 4800.0).unsqueeze(1)
            wall_factor = torch.where(wall_x | wall_y, torch.tensor(1.5, device=self.device),
                                      torch.tensor(1.0, device=self.device))
            rewards += weights[2] * (touched * height_term * speed_term * wall_factor)

        # ── R4: PlayerBallProximityVelocity (closest on team only) ──
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

        # ── R6: DefensivePositioning (support role only) ──
        if weights[5] > 0:
            is_support = ~is_closest
            own_goal_y = torch.where(is_blue, torch.tensor(-5120.0, device=self.device),
                                     torch.tensor(5120.0, device=self.device))
            own_goal_3d = torch.zeros(E, 4, 3, device=self.device)
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

        # ── R8: DemoAttempt ──
        if weights[7] > 0:
            opp0_pos = s.car_pos[:, ENEMY0_IDX]
            opp1_pos = s.car_pos[:, ENEMY1_IDX]
            to_opp0 = opp0_pos - s.car_pos
            to_opp1 = opp1_pos - s.car_pos
            d0 = to_opp0.norm(dim=-1)
            d1 = to_opp1.norm(dim=-1)
            nearest_dist = torch.min(d0, d1)
            nearest_vec = torch.where((d0 <= d1).unsqueeze(-1), to_opp0, to_opp1)
            nearest_dir = nearest_vec / (nearest_dist.unsqueeze(-1) + 1e-6)
            speed_to_opp = (s.car_vel * nearest_dir).sum(dim=-1)
            rewards += weights[7] * (
                torch.exp(-nearest_dist / 500.0) *
                (speed_to_opp / 2300.0).clamp(min=0) *
                (player_speed > 1500.0).float()
            )

        # ── R9: AirControl = max(dribble, aerial) ──
        if weights[8] > 0:
            bz = s.ball_pos[:, 2].unsqueeze(1)  # (E, 1)
            cz = s.car_pos[:, :, 2]              # (E, 4)
            ball_above = bz > (cz + 60.0)

            xy_diff = s.ball_pos[:, :2].unsqueeze(1) - s.car_pos[:, :, :2]
            xy_dist = xy_diff.norm(dim=-1)
            close_overhead = xy_dist < 180.0
            prox = (1.0 - xy_dist / 180.0).clamp(0, 1)
            ht = ((bz - cz - 60.0) / 250.0).clamp(0, 1)

            on_ground_f = (s.car_on_ground > 0.5).float()
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

        # ── Event Reward ──
        current_ev = torch.zeros(E, 4, 8, device=self.device)
        current_ev[:, :, 0] = s.match_goals
        for j in range(4):
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

        # ── Team Spirit blending ──
        if ts > 0:
            blue_mean = (rewards[:, 0] + rewards[:, 1]) * 0.5
            orange_mean = (rewards[:, 2] + rewards[:, 3]) * 0.5
            team_mean = torch.stack([blue_mean, blue_mean, orange_mean, orange_mean], dim=1)
            rewards = (1.0 - ts) * rewards + ts * team_mean

        # ── Update tracking state ──
        self.prev_player_boost.copy_(s.car_boost)

        return rewards
