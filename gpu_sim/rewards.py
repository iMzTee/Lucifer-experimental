"""rewards.py — Potential-based shaping + staged curriculum rewards on GPU.

Supports variable n_agents (1v0, 1v1, 2v2).
Operates on TensorState directly (no data movement).

Reward channels (11 continuous, 5 event):
  P1: BallToGoalPotential (potential-based)
  P2: VelTowardGoalPotential (potential-based)
  R3: TouchQuality (event-like, with consecutive air touch multiplier)
  R4: FaceBall (continuous, closest player)
  R5: SpeedTowardBall (continuous, closest player)
  R6: KickoffSpeed (conditional)
  R7: BoostPickup (event-like)
  R8: DefensivePos (conditional, support players)
  R9: AggressionBias (flat constant, stages 3+)
  R10: AirReward (binary airborne signal)
  R11: BoostConservation (sqrt boost amount)

Research basis: Necto (potential shaping, team_spirit), Lucy-SKG,
ZealanL PPO Guide (staged curriculum), Kaiyo-bot (consecutive air touches).
"""

import torch
from .constants import (
    BALL_MAX_SPEED, CAR_MAX_SPEED, BACK_NET_Y,
    GOAL_HEIGHT, BLUE_GOAL_BACK, ORANGE_GOAL_BACK,
)

# ── Reward weights per stage ──
# [P1:BallGoalPot, P2:VelGoalPot, R3:TouchQual, R4:FaceBall, R5:SpeedToBall,
#  R6:Kickoff, R7:BoostPickup, R8:DefPos, R9:AggrBias, R10:Air, R11:BoostSave]
STAGE_WEIGHTS = {
    0: [0.0, 0.0, 5.0, 1.0, 2.5, 3.0, 0.5, 0.0, 0.0, 0.0,  0.05],
    1: [2.0, 1.5, 4.0, 1.5, 1.5, 2.0, 0.5, 0.0, 0.0, 0.0,  0.1],
    2: [3.0, 2.0, 3.0, 1.0, 1.0, 2.0, 0.5, 0.0, 0.0, 0.2,  0.2],
    3: [4.0, 3.0, 3.0, 0.5, 0.5, 2.0, 1.0, 0.5, 0.2, 0.15, 0.25],
    4: [5.0, 3.5, 3.0, 0.3, 0.3, 1.5, 1.0, 1.0, 0.2, 0.1,  0.25],
    5: [5.0, 3.5, 3.0, 0.2, 0.2, 1.5, 1.0, 1.5, 0.2, 0.1,  0.25],
}

# [GoalScored, TeamScore, OppScore, Demo, Touched]
EVENT_WEIGHTS = {
    0: [ 0.0,  0.0,   0.0, 0.0, 3.0],  # NO goal rewards. Strong touch.
    1: [ 5.0,  3.0,  -5.0, 0.0, 0.3],  # Light goal rewards
    2: [ 8.0,  5.0,  -7.0, 0.0, 0.2],  # Moderate goal rewards
    3: [10.0,  7.0,  -8.0, 3.0, 0.1],  # Full goal + demo rewards
    4: [12.0,  8.0, -10.0, 5.0, 0.0],  # Strong goal emphasis
    5: [12.0,  8.0, -10.0, 5.0, 0.0],  # Same for 2v2
}

TEAM_SPIRIT = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.6}

# Potential discount factor (gamma for shaping)
GAMMA = 0.99


class GPURewards:
    """Computes reward signals for all envs/agents in batched PyTorch.

    Supports n_agents = 1, 2, or 4.
    """

    def __init__(self, n_envs, device='cuda', n_agents=4, layout=None):
        self.n_envs = n_envs
        self.n_agents = n_agents
        self.device = device
        self.layout = layout

        # Event tracking: [goals, team_score, opp_score, demos, touched]
        self.event_last = torch.zeros(n_envs, n_agents, 5, device=device)

        # Kickoff state
        self.is_kickoff = torch.zeros(n_envs, dtype=torch.bool, device=device)

        # Previous boost for R7
        self.prev_player_boost = torch.zeros(n_envs, n_agents, device=device)

        # Potential-based shaping state
        self.prev_phi_ball_pos = torch.zeros(n_envs, n_agents, device=device)
        self.prev_phi_ball_vel = torch.zeros(n_envs, n_agents, device=device)

        # Consecutive air touch tracking (Kaiyo-bot)
        self.consecutive_air_touches = torch.zeros(n_envs, n_agents, device=device)

        # Pre-compute goal positions for broadcasting
        self._orange_goal = ORANGE_GOAL_BACK.to(device)
        self._blue_goal = BLUE_GOAL_BACK.to(device)

        # Pre-allocate weight tensors
        self._stage_weights = {}
        self._event_weights = {}
        for s in range(6):
            self._stage_weights[s] = torch.tensor(STAGE_WEIGHTS[s], device=device)
            self._event_weights[s] = torch.tensor(EVENT_WEIGHTS[s], device=device)

    def _compute_potentials(self, ball_pos, ball_vel, opp_goal, is_blue_f):
        """Compute phi values for potential-based shaping.

        Returns: (phi_ball_pos, phi_ball_vel) each (E, A)
        """
        ball_e = ball_pos.unsqueeze(1)  # (E, 1, 3)

        # P1: Ball position potential
        dist_opp = (ball_e - opp_goal).norm(dim=-1)  # (E, A)
        own_goal = (is_blue_f.unsqueeze(-1) * self._blue_goal +
                    (1 - is_blue_f).unsqueeze(-1) * self._orange_goal)
        dist_own = (ball_e - own_goal).norm(dim=-1)  # (E, A)
        phi_pos = torch.exp(-dist_opp / 5000.0) - 0.5 * torch.exp(-dist_own / 5000.0)

        # P2: Ball velocity toward opponent goal potential
        dir_to_opp = opp_goal - ball_e  # (E, A, 3)
        dir_to_opp_n = dir_to_opp / (dir_to_opp.norm(dim=-1, keepdim=True) + 1e-6)
        ball_vel_e = ball_vel.unsqueeze(1)  # (E, 1, 3)
        phi_vel = (ball_vel_e * dir_to_opp_n).sum(dim=-1) / BALL_MAX_SPEED  # (E, A)

        return phi_pos, phi_vel

    def reset_envs(self, mask, state):
        """Reset reward state for terminated envs. mask: (E,) bool."""
        if not mask.any():
            return

        s = state
        is_blue = s.car_team == 0  # (E, A)
        is_blue_f = is_blue.float()

        # Event baselines
        goals = s.match_goals[mask]
        touched = s.car_ball_touched[mask]
        demos = s.match_demos[mask]

        blue_mask_per_agent = is_blue[mask].float()
        team_score = (blue_mask_per_agent * s.blue_score[mask].unsqueeze(1).float() +
                      (1 - blue_mask_per_agent) * s.orange_score[mask].unsqueeze(1).float())
        opp_score = (blue_mask_per_agent * s.orange_score[mask].unsqueeze(1).float() +
                     (1 - blue_mask_per_agent) * s.blue_score[mask].unsqueeze(1).float())

        self.event_last[mask, :, 0] = goals
        self.event_last[mask, :, 1] = team_score
        self.event_last[mask, :, 2] = opp_score
        self.event_last[mask, :, 3] = demos
        self.event_last[mask, :, 4] = touched

        # Kickoff detection
        bp = s.ball_pos[mask]
        self.is_kickoff[mask] = (
            (torch.abs(bp[:, 0]) < 50) &
            (torch.abs(bp[:, 1]) < 50) &
            (bp[:, 2] < 120)
        )

        self.prev_player_boost[mask] = s.car_boost[mask]

        # Initialize potentials from post-reset state
        opp_goal_m = (is_blue_f[mask].unsqueeze(-1) * self._orange_goal +
                      (1 - is_blue_f[mask]).unsqueeze(-1) * self._blue_goal)
        phi_pos, phi_vel = self._compute_potentials(
            s.ball_pos[mask], s.ball_vel[mask], opp_goal_m, is_blue_f[mask]
        )
        self.prev_phi_ball_pos[mask] = phi_pos
        self.prev_phi_ball_vel[mask] = phi_vel

        # Reset air touch combos
        self.consecutive_air_touches[mask] = 0

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

        is_blue = (s.car_team == 0)
        is_blue_f = is_blue.float()

        # Goal positions per agent
        opp_goal = (is_blue_f.unsqueeze(-1) * self._orange_goal +
                    (1 - is_blue_f).unsqueeze(-1) * self._blue_goal)

        # Closest on team
        if A <= 2:
            is_closest = torch.ones(E, A, dtype=torch.bool, device=self.device)
        else:
            ally_idx = layout["ally_idx"]
            ally_dist = torch.stack([to_ball_dist[:, ally_idx[i]] for i in range(A)], dim=1)
            is_closest = to_ball_dist <= ally_dist

        ball_speed = s.ball_vel.norm(dim=-1)  # (E,)

        rewards = torch.zeros(E, A, device=self.device)

        # ── P1: BallToGoalPotential (potential-based shaping) ──
        # phi = exp(-dist_opp/5000) - 0.5*exp(-dist_own/5000)
        # reward = gamma*phi(s') - phi(s)
        phi_pos, phi_vel = self._compute_potentials(
            s.ball_pos, s.ball_vel, opp_goal, is_blue_f
        )

        if weights[0] > 0:
            rewards += weights[0] * (GAMMA * phi_pos - self.prev_phi_ball_pos)

        # ── P2: VelTowardGoalPotential (potential-based shaping) ──
        if weights[1] > 0:
            rewards += weights[1] * (GAMMA * phi_vel - self.prev_phi_ball_vel)

        # Update potential state
        self.prev_phi_ball_pos.copy_(phi_pos)
        self.prev_phi_ball_vel.copy_(phi_vel)

        # ── R3: TouchQuality ──
        if weights[2] > 0:
            touched = s.car_ball_touched
            ball_z = s.ball_pos[:, 2].unsqueeze(1)
            ball_speed_e = ball_speed.unsqueeze(1)

            # Height: sqrt scaling (Necto approach), only for aerial touches
            height_mult = 1.0 + torch.sqrt(ball_z.clamp(min=0) / 2044.0)
            height_mult = torch.where(s.car_on_ground < 0.5, height_mult, torch.ones_like(height_mult))

            # Speed: normalize by BALL_MAX_SPEED
            speed_factor = 0.5 + 0.5 * (ball_speed_e / BALL_MAX_SPEED).clamp(max=2.0)

            # Wall bonus
            wall_x = (torch.abs(s.ball_pos[:, 0]) > 3800.0).unsqueeze(1)
            wall_y = (torch.abs(s.ball_pos[:, 1]) > 4800.0).unsqueeze(1)
            wall_factor = torch.where(wall_x | wall_y, torch.tensor(1.5, device=self.device),
                                      torch.tensor(1.0, device=self.device))

            # Consecutive air touch multiplier (Kaiyo-bot)
            # Update air touch tracking
            air_touch = (touched > 0.5) & (s.car_on_ground < 0.5)
            on_ground = s.car_on_ground > 0.5
            count = self.consecutive_air_touches
            self.consecutive_air_touches = torch.where(
                air_touch, count + 1,
                torch.where(on_ground, torch.zeros_like(count), count)
            )
            air_touch_mult = 1.0 + self.consecutive_air_touches.clamp(max=5).pow(1.4)
            # Only apply multiplier when actually touching in air; otherwise mult=1
            effective_mult = torch.where(air_touch, air_touch_mult, torch.ones_like(air_touch_mult))

            rewards += weights[2] * (touched * height_mult * speed_factor * wall_factor * effective_mult)

        # ── R4: FaceBall (distance-gated: fades to 0 within 500uu) ──
        if weights[3] > 0:
            facing_ball = (s.car_fwd * to_ball_dir).sum(dim=-1).clamp(min=0)
            dist_gate = (to_ball_dist / 500.0).clamp(max=1.0)  # 0 at ball, 1 at 500+
            rewards += weights[3] * (facing_ball * dist_gate * is_closest.float())

        # ── R5: SpeedTowardBall (distance-gated: fades to 0 within 500uu) ──
        if weights[4] > 0:
            speed_toward_ball = (s.car_vel * to_ball_dir).sum(dim=-1) / CAR_MAX_SPEED
            speed_toward_ball = speed_toward_ball.clamp(min=0)
            dist_gate = (to_ball_dist / 500.0).clamp(max=1.0)  # 0 at ball, 1 at 500+
            rewards += weights[4] * (speed_toward_ball * dist_gate * is_closest.float())

        # ── R6: KickoffSpeed ──
        if weights[5] > 0:
            self.is_kickoff = self.is_kickoff & (ball_speed < 100)
            kick_speed = (s.car_vel * to_ball_dir).sum(dim=-1)
            kick_r = (kick_speed / CAR_MAX_SPEED).clamp(min=0) + torch.exp(-to_ball_dist / 800.0)
            rewards += weights[5] * (kick_r * self.is_kickoff.unsqueeze(1).float())
        else:
            self.is_kickoff = self.is_kickoff & (ball_speed < 100)

        # ── R7: BoostPickup ──
        if weights[6] > 0:
            boost_gained = (s.car_boost - self.prev_player_boost).clamp(min=0)
            is_small = (boost_gained > 0.01) & (boost_gained <= 0.15)
            pad_mult = torch.where(is_small, torch.tensor(2.0, device=self.device),
                                   torch.tensor(1.0, device=self.device))
            rewards += weights[6] * (torch.sqrt(boost_gained) * pad_mult).clamp(max=0.5)

        # ── R8: DefensivePos (skip for 1v0, support players only) ──
        if weights[7] > 0 and A >= 2:
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
            rewards += weights[7] * (is_support.float() * align * gaussian)

        # ── R9: AggressionBias (flat constant, stages 3+) ──
        if weights[8] > 0:
            rewards += weights[8]

        # ── R10: AirReward (binary airborne signal) ──
        if weights[9] > 0:
            rewards += weights[9] * (s.car_on_ground < 0.5).float()

        # ── R11: BoostConservation ──
        # Reward having boost (sqrt shape) + penalize wasteful usage
        # Wasteful = boost spent (delta < 0) without gaining speed toward ball
        if weights[10] > 0:
            # Base: reward for current boost level
            boost_reward = torch.sqrt(s.car_boost)

            # Penalty: boost spent without productive speed gain
            boost_spent = (self.prev_player_boost - s.car_boost).clamp(min=0)  # (E, A)
            speed_to_ball = (s.car_vel * to_ball_dir).sum(dim=-1) / CAR_MAX_SPEED  # (E, A)
            speed_gained = speed_to_ball.clamp(min=0)
            # If spending boost but not moving toward ball effectively, penalize
            waste_penalty = boost_spent * (1.0 - speed_gained).clamp(min=0)

            rewards += weights[10] * (boost_reward - 0.5 * waste_penalty)

        # ── Event Reward ──
        current_ev = torch.zeros(E, A, 5, device=self.device)
        current_ev[:, :, 0] = s.match_goals
        for j in range(A):
            blue_mask = (s.car_team[:, j] == 0).float()
            current_ev[:, j, 1] = blue_mask * s.blue_score.float() + (1 - blue_mask) * s.orange_score.float()
            current_ev[:, j, 2] = blue_mask * s.orange_score.float() + (1 - blue_mask) * s.blue_score.float()
        current_ev[:, :, 3] = s.match_demos
        current_ev[:, :, 4] = s.car_ball_touched

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

        return rewards
