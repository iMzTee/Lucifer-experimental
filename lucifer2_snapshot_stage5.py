import os
import time
import json
import torch
import numpy as np
import multiprocessing
import warnings
import sys
import psutil
from torch.utils.tensorboard import SummaryWriter

# --- FILE LOGGING ---
# Tee stdout/stderr to lucifer_training.log so metrics survive os.execv restarts.
# Only active in the main process — workers use multiprocessing spawn and must not
# compete over the log file (Windows file locking would silently drop output).
class _TeeOutput:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data)
            except: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except: pass
    def fileno(self):
        return self.streams[0].fileno()

if multiprocessing.current_process().name == "MainProcess":
    _log_file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lucifer_training.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeOutput(sys.__stdout__, _log_file)
    sys.stderr = _TeeOutput(sys.__stderr__, _log_file)

# --- CORE IMPORTS ---
import rlgym_sim
from rlgym_sim.utils.reward_functions.combined_reward import CombinedReward
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition
from rlgym_sim.utils.reward_functions.common_rewards import (
    VelocityBallToGoalReward, EventReward, TouchBallReward, VelocityPlayerToBallReward
)
from rlgym_sim.utils.action_parsers.discrete_act import DiscreteAction
from rlgym_sim.utils.state_setters import StateSetter, DefaultState

import rlgym_ppo.ppo
import rlgym_ppo.batched_agents
import rlgym_ppo.util as rlgym_util

# --- CURRICULUM STATE ---
# Persists training state (stage, hyperparams, rolling metrics) across restarts.
# When a stage advance or hyperparam change is needed, the process saves this file,
# saves a checkpoint, then calls os.execv to restart itself cleanly.
CURRICULUM_STATE_FILE = "curriculum_state.json"
CURRICULUM_STAGE_NAMES = ["Foundations", "Awareness", "Game Sense", "Teamplay", "Aerial", "Dribbling", "Mastery"]

DEFAULT_CURRICULUM_STATE = {
    "stage": 0,
    "policy_lr": 2e-4,
    "critic_lr": 4e-4,
    "ent_coef": 0.01,
    "reward_window": [],       # rolling mean rewards (training iters only)
    "clip_high_count": 0,      # consecutive training iters with clip fraction > threshold
    "stage_iter_count": 0,     # training iters elapsed since last stage advance
    "value_loss_window": [],   # rolling value loss (training iters only)
    "value_loss_high_count": 0,  # consecutive iters with value loss above high threshold
}

def load_curriculum_state():
    if os.path.exists(CURRICULUM_STATE_FILE):
        state = {}
        with open(CURRICULUM_STATE_FILE) as f:
            state = json.load(f)
        for k, v in DEFAULT_CURRICULUM_STATE.items():
            if k not in state:
                state[k] = v
        return state
    return dict(DEFAULT_CURRICULUM_STATE)

def save_curriculum_state(state):
    with open(CURRICULUM_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def kill_zombies():
    """Kills only direct child worker processes spawned by this script."""
    print("\n[*] SURGICAL PURGE OF WORKER PROCESSES...")
    current_pid = os.getpid()
    try:
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        gone, still_alive = psutil.wait_procs(children, timeout=3)
        if still_alive:
            print(f"[!] {len(still_alive)} worker(s) didn't die — forcing...")
            for p in still_alive:
                try: p.kill()
                except: pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    print("[*] WORKERS PURGED.")


# ---------------------------------------------------------------------------
# REWARD CLASSES
# ---------------------------------------------------------------------------

class MovementConsistencyReward(rlgym_sim.utils.reward_functions.RewardFunction):
    # Reward forward speed, penalize idling. Teaches the bot to always be moving.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        vel = player.car_data.linear_velocity
        fwd = player.car_data.forward
        if callable(fwd): fwd = fwd()
        forward_speed = np.dot(vel, fwd)
        idle_penalty = -0.5 if np.linalg.norm(vel) < 50 else 0
        return max(-1.0, (forward_speed / 2300.0) + idle_penalty)

class FacingBallReward(rlgym_sim.utils.reward_functions.RewardFunction):
    # Reward pointing toward the ball. Teaches the bot to orient before committing.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        pos_diff = state.ball.position - player.car_data.position
        pos_diff /= (np.linalg.norm(pos_diff) + 1e-6)
        fwd = player.car_data.forward
        if callable(fwd): fwd = fwd()
        return max(0.0, float(np.dot(fwd, pos_diff)))

class BallProximityReward(rlgym_sim.utils.reward_functions.RewardFunction):
    # Reward closing distance to the ball. Exponential falloff: ~1.0 touching, ~0.08 at max range.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        dist = np.linalg.norm(state.ball.position - player.car_data.position)
        return np.exp(-dist / 2000.0)

class BallFleeingPenalty(rlgym_sim.utils.reward_functions.RewardFunction):
    # Penalize moving away from the ball when within striking range (~2000 units).
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        dist = np.linalg.norm(state.ball.position - player.car_data.position)
        if dist > 2000.0:
            return 0.0
        to_ball = state.ball.position - player.car_data.position
        to_ball /= (np.linalg.norm(to_ball) + 1e-6)
        vel = player.car_data.linear_velocity
        vel_norm = np.linalg.norm(vel)
        if vel_norm < 1e-6:
            return 0.0
        moving_away = np.dot(vel / vel_norm, -to_ball)
        return -max(0.0, moving_away)

class BoostHoardingPenalty(rlgym_sim.utils.reward_functions.RewardFunction):
    # Penalize sitting at full boost without using it.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        if player.boost_amount >= 1.0:
            return -0.3
        return 0.0

class BallGoalDangerPenalty(rlgym_sim.utils.reward_functions.RewardFunction):
    # Penalize when ball is near own goal and bot is far from it. Teaches defensive urgency.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        own_goal_y = -5120.0 if player.team_num == 0 else 5120.0
        own_goal_pos = np.array([0.0, own_goal_y, 0.0])
        ball_to_goal = np.linalg.norm(state.ball.position - own_goal_pos)
        if ball_to_goal > 2500.0:
            return 0.0
        player_to_goal = np.linalg.norm(player.car_data.position - own_goal_pos)
        return -min(1.0, player_to_goal / 5000.0)

class OpponentPressureReward(rlgym_sim.utils.reward_functions.RewardFunction):
    # Reward pressuring the opponent who has possession of the ball.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        our_dist = np.linalg.norm(state.ball.position - player.car_data.position)
        closest_opp = None
        closest_opp_dist = float('inf')
        for p in state.players:
            if p.team_num != player.team_num:
                d = np.linalg.norm(state.ball.position - p.car_data.position)
                if d < closest_opp_dist:
                    closest_opp_dist = d
                    closest_opp = p
        if closest_opp is None or closest_opp_dist >= our_dist:
            return 0.0
        dist_to_opp = np.linalg.norm(player.car_data.position - closest_opp.car_data.position)
        return np.exp(-dist_to_opp / 1500.0)

class OffensivePositioningReward(rlgym_sim.utils.reward_functions.RewardFunction):
    # Reward being between the ball and opponent's goal when near the ball.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        opp_goal_y = 5120.0 if player.team_num == 0 else -5120.0
        opp_goal_pos = np.array([0.0, opp_goal_y, 0.0])
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        dist_to_ball = np.linalg.norm(ball_pos - player_pos)
        if dist_to_ball > 3000.0:
            return 0.0
        ball_to_goal = opp_goal_pos - ball_pos
        ball_to_player = player_pos - ball_pos
        ball_to_goal_norm = np.linalg.norm(ball_to_goal) + 1e-6
        alignment = np.dot(ball_to_player, ball_to_goal) / (ball_to_goal_norm * (np.linalg.norm(ball_to_player) + 1e-6))
        return max(0.0, float(alignment))

class DefensiveShadowingReward(rlgym_sim.utils.reward_functions.RewardFunction):
    # Reward staying between ball and own goal when teammate is challenging.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        own_goal_y = -5120.0 if player.team_num == 0 else 5120.0
        own_goal_pos = np.array([0.0, own_goal_y, 0.0])
        ball_pos = state.ball.position
        player_pos = player.car_data.position
        our_dist_to_ball = np.linalg.norm(ball_pos - player_pos)
        for p in state.players:
            if p.team_num == player.team_num and p is not player:
                teammate_dist = np.linalg.norm(ball_pos - p.car_data.position)
                if teammate_dist < our_dist_to_ball:
                    goal_to_ball = ball_pos - own_goal_pos
                    goal_to_player = player_pos - own_goal_pos
                    g2b_norm = np.linalg.norm(goal_to_ball) + 1e-6
                    alignment = np.dot(goal_to_player, goal_to_ball) / (g2b_norm * (np.linalg.norm(goal_to_player) + 1e-6))
                    dist_to_line = np.linalg.norm(goal_to_player) / g2b_norm
                    if 0.0 < dist_to_line < 1.2:
                        return max(0.0, float(alignment))
        return 0.0

class TeammateBumpPenalty(rlgym_sim.utils.reward_functions.RewardFunction):
    # Penalize high-speed proximity to teammates (proxy for accidental bumps).
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        player_vel = player.car_data.linear_velocity
        for p in state.players:
            if p.team_num == player.team_num and p is not player:
                dist = np.linalg.norm(player.car_data.position - p.car_data.position)
                if dist < 300.0:
                    rel_vel = np.linalg.norm(player_vel - p.car_data.linear_velocity)
                    if rel_vel > 500.0:
                        return -min(1.0, rel_vel / 2300.0)
        return 0.0

_LARGE_BOOST_PAD_POSITIONS = np.array([
    [-3584.0,    0.0, 73.0],
    [ 3584.0,    0.0, 73.0],
    [-3072.0,  4096.0, 73.0],
    [ 3072.0,  4096.0, 73.0],
    [-3072.0, -4096.0, 73.0],
    [ 3072.0, -4096.0, 73.0],
], dtype=np.float32)

class BoostManagementReward(rlgym_sim.utils.reward_functions.RewardFunction):
    # Reward picking up boost when low; penalize wasting it when fast and full.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        boost = player.boost_amount
        speed = np.linalg.norm(player.car_data.linear_velocity)
        reward = 0.0
        if boost < 0.3:
            dists = np.linalg.norm(_LARGE_BOOST_PAD_POSITIONS - player.car_data.position, axis=1)
            if dists.min() < 500.0:
                reward += 0.5
        if boost > 0.8 and speed > 2000.0:
            reward -= 0.2
        return reward

class TeammateBoostRespectPenalty(rlgym_sim.utils.reward_functions.RewardFunction):
    # Penalize heading toward a boost pad that a closer teammate is already committing to.
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        player_pos = player.car_data.position
        player_vel = player.car_data.linear_velocity
        player_speed = np.linalg.norm(player_vel) + 1e-6
        for pad_pos in _LARGE_BOOST_PAD_POSITIONS:
            dist_to_pad = np.linalg.norm(pad_pos - player_pos)
            if dist_to_pad > 1500.0:
                continue
            to_pad = (pad_pos - player_pos)
            to_pad_norm = np.linalg.norm(to_pad) + 1e-6
            our_alignment = np.dot(player_vel / player_speed, to_pad / to_pad_norm)
            if our_alignment < 0.7:
                continue
            for p in state.players:
                if p.team_num != player.team_num or p is player:
                    continue
                teammate_dist = np.linalg.norm(pad_pos - p.car_data.position)
                if teammate_dist >= dist_to_pad:
                    continue
                t_vel = p.car_data.linear_velocity
                t_speed = np.linalg.norm(t_vel) + 1e-6
                t_to_pad = (pad_pos - p.car_data.position)
                t_to_pad_norm = np.linalg.norm(t_to_pad) + 1e-6
                t_alignment = np.dot(t_vel / t_speed, t_to_pad / t_to_pad_norm)
                if t_alignment > 0.7:
                    return -0.5
        return 0.0


class DribbleReward(rlgym_sim.utils.reward_functions.RewardFunction):
    """Reward for ball resting on top of car — teaches dribbling control."""
    def reset(self, state): pass
    def get_reward(self, player, state, previous_action) -> float:
        ball = state.ball.position
        car  = player.car_data.position
        if ball[2] < car[2] + 60:        # ball must be above car
            return 0.0
        xy_dist = np.linalg.norm(ball[:2] - car[:2])
        if xy_dist > 180.0:              # ball must be close overhead
            return 0.0
        proximity = 1.0 - xy_dist / 180.0
        height    = min(1.0, (ball[2] - car[2] - 60) / 250.0)
        return proximity * (0.3 + 0.7 * height)


class AerialStateSetter(StateSetter):
    """Mixes ground kickoffs and aerial ball spawns. Higher prob = more aerial training."""
    _KICKOFF_POS = np.array([
        [-2048, -2560, 17], [2048, -2560, 17],
        [-256,  -3840, 17], [256,  -3840, 17], [0, -4608, 17],
    ], dtype=np.float32)

    def __init__(self, aerial_prob=0.5):
        self.aerial_prob = aerial_prob

    def reset(self, state_wrapper):
        # --- Ball ---
        if np.random.random() < self.aerial_prob:
            z   = np.random.uniform(400, 1600)
            zvl = np.random.uniform(-200, 200)
            state_wrapper.ball.set_pos(
                np.random.uniform(-2500, 2500),
                np.random.uniform(-2500, 2500), z)
            state_wrapper.ball.set_lin_vel(
                np.random.uniform(-400, 400),
                np.random.uniform(-400, 400), zvl)
        else:
            state_wrapper.ball.set_pos(0, 0, 93)
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        state_wrapper.ball.set_ang_vel(0, 0, 0)

        # --- Cars ---
        n_blue = len(state_wrapper.blue_cars())
        n_org  = len(state_wrapper.orange_cars())
        idxs_b = np.random.choice(len(self._KICKOFF_POS), n_blue,  replace=False)
        idxs_o = np.random.choice(len(self._KICKOFF_POS), n_org,   replace=False)

        for i, car in enumerate(state_wrapper.blue_cars()):
            p = self._KICKOFF_POS[idxs_b[i]]
            car.set_pos(p[0], p[1], p[2])
            car.set_rot(0, np.pi / 2, 0)   # face +Y (orange goal)
            car.set_lin_vel(0, 0, 0)
            car.boost = np.random.uniform(0.5, 1.0)  # extra boost for aerials

        for i, car in enumerate(state_wrapper.orange_cars()):
            p = self._KICKOFF_POS[idxs_o[i]]
            car.set_pos(-p[0], -p[1], p[2])  # mirror X,Y; keep Z
            car.set_rot(0, -np.pi / 2, 0)   # face -Y (blue goal)
            car.set_lin_vel(0, 0, 0)
            car.boost = np.random.uniform(0.5, 1.0)


# ---------------------------------------------------------------------------
# CURRICULUM STAGES
# ---------------------------------------------------------------------------
#
# Stage 0 — Foundations: move, face ball, approach, touch, don't flee.
#   Advance: rolling avg reward > 1.20 for 20 training iters, or 60 training iters elapsed.
#
# Stage 1 — Awareness: + defensive urgency, boost discipline, stronger goal-direction.
#   Bot reliably touches ball. Now teach situational awareness.
#   Advance: rolling avg reward > 1.12 for 20 training iters, or 80 training iters elapsed.
#
# Stage 2 — Game Sense: + opponent pressure, offensive positioning, boost management.
#   Bot has awareness. Now teach reading the game.
#   Advance: rolling avg reward > 1.18 for 20 training iters, or 100 training iters elapsed.
#
# Stage 3 — Teamplay: full reward set including defensive shadowing and team coordination.
#   Final stage, no advancement.
#
# Entropy notes:
#   Ideal range is 3–5. Early training entropy will be 6–7 (chaotic exploration) and should
#   fall naturally. Do NOT intervene while entropy > 3.0. Only act if it collapses below 3.0.
#
# ---------------------------------------------------------------------------

def env_factory():
    _original_excepthook = sys.excepthook
    def _quiet_excepthook(exc_type, exc_val, exc_tb):
        if exc_type is KeyboardInterrupt:
            return
        _original_excepthook(exc_type, exc_val, exc_tb)
    sys.excepthook = _quiet_excepthook

    stage = load_curriculum_state()["stage"]

    if stage == 0:
        # Foundations: teach the bot to move, face the ball, get to it, and touch it.
        # No positioning or strategy yet — that requires knowing what a good touch is first.
        rewards = [
            MovementConsistencyReward(),  # stay moving, don't idle
            FacingBallReward(),           # orient toward ball
            BallProximityReward(),        # get close to ball
            BallFleeingPenalty(),         # don't bounce away after contact
            VelocityBallToGoalReward(),   # faint goal-direction hint
            EventReward(goal=5.0, touch=2.0, demo=-0.5),
        ]
        weights = [
            0.5,   # MovementConsistencyReward
            1.0,   # FacingBallReward
            3.0,   # BallProximityReward        — primary signal
            0.8,   # BallFleeingPenalty
            0.2,   # VelocityBallToGoalReward   — aspirational only
            1.0,   # EventReward
        ]

    elif stage == 1:
        # Awareness: bot reliably approaches and touches ball.
        # Add defensive urgency and basic boost discipline.
        # Increase goal-direction weight now that the bot can actually hit the ball.
        rewards = [
            MovementConsistencyReward(),
            FacingBallReward(),
            BallProximityReward(),
            BallFleeingPenalty(),
            VelocityBallToGoalReward(),
            BallGoalDangerPenalty(),   # react when ball is near own goal
            BoostHoardingPenalty(),    # don't sit on full boost
            EventReward(goal=6.0, touch=1.5, demo=-0.5),
        ]
        weights = [
            0.5,   # MovementConsistencyReward
            1.0,   # FacingBallReward
            2.5,   # BallProximityReward
            0.8,   # BallFleeingPenalty
            0.5,   # VelocityBallToGoalReward   — now meaningful
            1.0,   # BallGoalDangerPenalty
            0.5,   # BoostHoardingPenalty
            1.0,   # EventReward
        ]

    elif stage == 2:
        # Game Sense: bot has basic awareness. Teach reading the game:
        # pressure opponents, get into shooting positions, manage boost intelligently.
        rewards = [
            MovementConsistencyReward(),
            FacingBallReward(),
            BallProximityReward(),
            BallFleeingPenalty(),
            VelocityBallToGoalReward(),
            BallGoalDangerPenalty(),
            BoostHoardingPenalty(),
            OpponentPressureReward(),      # disrupt opponent possession
            OffensivePositioningReward(),  # shoot from good angles
            BoostManagementReward(),       # collect boost intelligently
            EventReward(goal=7.0, touch=1.0, demo=-0.5),
        ]
        weights = [
            0.4,   # MovementConsistencyReward
            0.8,   # FacingBallReward
            2.0,   # BallProximityReward
            0.8,   # BallFleeingPenalty
            0.6,   # VelocityBallToGoalReward
            1.0,   # BallGoalDangerPenalty
            0.5,   # BoostHoardingPenalty
            0.8,   # OpponentPressureReward
            0.7,   # OffensivePositioningReward
            0.5,   # BoostManagementReward
            1.0,   # EventReward
        ]

    elif stage == 3:
        # Teamplay: full reward set. Bot understands the game; teach team coordination.
        rewards = [
            MovementConsistencyReward(),
            FacingBallReward(),
            BallProximityReward(),
            BallFleeingPenalty(),
            VelocityBallToGoalReward(),
            BallGoalDangerPenalty(),
            BoostHoardingPenalty(),
            OpponentPressureReward(),
            OffensivePositioningReward(),
            BoostManagementReward(),
            DefensiveShadowingReward(),        # hold defensive shape
            TeammateBumpPenalty(),             # don't bump teammates
            TeammateBoostRespectPenalty(),     # don't steal teammate's boost
            EventReward(goal=7.0, touch=1.0, demo=-0.5),
        ]
        weights = [
            0.4,   # MovementConsistencyReward
            0.8,   # FacingBallReward
            2.0,   # BallProximityReward
            0.8,   # BallFleeingPenalty
            0.6,   # VelocityBallToGoalReward
            1.0,   # BallGoalDangerPenalty
            0.5,   # BoostHoardingPenalty
            0.8,   # OpponentPressureReward
            0.7,   # OffensivePositioningReward
            0.5,   # BoostManagementReward
            0.7,   # DefensiveShadowingReward
            0.8,   # TeammateBumpPenalty
            0.6,   # TeammateBoostRespectPenalty
            1.0,   # EventReward
        ]

    elif stage == 4:
        # Aerial: introduce aerial ball spawns + large aerial touch bonus.
        # Primary new signal: TouchBallReward(aerial_weight=2.0) spikes for ceiling-height touches.
        rewards = [
            MovementConsistencyReward(),
            FacingBallReward(),
            BallProximityReward(),
            BallFleeingPenalty(),
            VelocityBallToGoalReward(),
            BallGoalDangerPenalty(),
            BoostHoardingPenalty(),
            OpponentPressureReward(),
            OffensivePositioningReward(),
            BoostManagementReward(),
            DefensiveShadowingReward(),
            TeammateBumpPenalty(),
            TeammateBoostRespectPenalty(),
            TouchBallReward(aerial_weight=2.0),  # KEY: huge signal for aerial touches
            VelocityPlayerToBallReward(),        # chase ball directionally in 3D
            EventReward(goal=7.0, touch=1.5, shot=3.0, demo=-0.5),
        ]
        weights = [
            0.3,   # MovementConsistencyReward
            0.6,   # FacingBallReward
            1.5,   # BallProximityReward
            0.6,   # BallFleeingPenalty
            0.6,   # VelocityBallToGoalReward
            0.8,   # BallGoalDangerPenalty
            0.4,   # BoostHoardingPenalty
            0.6,   # OpponentPressureReward
            0.5,   # OffensivePositioningReward
            0.4,   # BoostManagementReward
            0.5,   # DefensiveShadowingReward
            0.6,   # TeammateBumpPenalty
            0.4,   # TeammateBoostRespectPenalty
            2.0,   # TouchBallReward
            0.5,   # VelocityPlayerToBallReward
            1.0,   # EventReward
        ]
        aerial_prob = 0.5

    elif stage == 5:
        # Dribbling: keep aerial training, add dribble-control reward.
        rewards = [
            MovementConsistencyReward(),
            FacingBallReward(),
            BallProximityReward(),
            BallFleeingPenalty(),
            VelocityBallToGoalReward(),
            BallGoalDangerPenalty(),
            BoostHoardingPenalty(),
            OpponentPressureReward(),
            OffensivePositioningReward(),
            BoostManagementReward(),
            DefensiveShadowingReward(),
            TeammateBumpPenalty(),
            TeammateBoostRespectPenalty(),
            TouchBallReward(aerial_weight=2.0),
            VelocityPlayerToBallReward(),
            DribbleReward(),                     # KEY: teach ball control on car
            EventReward(goal=8.0, touch=1.0, shot=4.0, save=2.0, demo=-0.5),
        ]
        weights = [
            0.2,   # MovementConsistencyReward
            0.5,   # FacingBallReward
            1.2,   # BallProximityReward
            0.5,   # BallFleeingPenalty
            0.7,   # VelocityBallToGoalReward
            0.8,   # BallGoalDangerPenalty
            0.4,   # BoostHoardingPenalty
            0.6,   # OpponentPressureReward
            0.5,   # OffensivePositioningReward
            0.4,   # BoostManagementReward
            0.5,   # DefensiveShadowingReward
            0.6,   # TeammateBumpPenalty
            0.4,   # TeammateBoostRespectPenalty
            2.0,   # TouchBallReward
            0.5,   # VelocityPlayerToBallReward
            1.5,   # DribbleReward
            1.0,   # EventReward
        ]
        aerial_prob = 0.6

    else:
        # Mastery (Stage 6+): outcome-focused — remove hand-holding proximity reward,
        # let the bot rely on intrinsic game sense. Outcome rewards dominate.
        rewards = [
            MovementConsistencyReward(),
            FacingBallReward(),
            BallFleeingPenalty(),
            VelocityBallToGoalReward(),
            BallGoalDangerPenalty(),
            BoostHoardingPenalty(),
            OpponentPressureReward(),
            OffensivePositioningReward(),
            BoostManagementReward(),
            DefensiveShadowingReward(),
            TeammateBumpPenalty(),
            TeammateBoostRespectPenalty(),
            TouchBallReward(aerial_weight=2.0),
            VelocityPlayerToBallReward(),
            DribbleReward(),
            EventReward(goal=10.0, touch=0.5, shot=5.0, save=3.0, demo=-0.5),
        ]
        weights = [
            0.2,   # MovementConsistencyReward
            0.5,   # FacingBallReward
            0.5,   # BallFleeingPenalty
            0.7,   # VelocityBallToGoalReward
            0.8,   # BallGoalDangerPenalty
            0.4,   # BoostHoardingPenalty
            0.6,   # OpponentPressureReward
            0.5,   # OffensivePositioningReward
            0.4,   # BoostManagementReward
            0.5,   # DefensiveShadowingReward
            0.6,   # TeammateBumpPenalty
            0.4,   # TeammateBoostRespectPenalty
            2.0,   # TouchBallReward
            0.5,   # VelocityPlayerToBallReward
            1.5,   # DribbleReward
            1.0,   # EventReward
        ]
        aerial_prob = 0.65

    state_setter = AerialStateSetter(aerial_prob) if stage >= 4 else DefaultState()
    return rlgym_sim.make(tick_skip=8, team_size=2, spawn_opponents=True,
                          terminal_conditions=[GoalScoredCondition(), TimeoutCondition(300)],
                          reward_fn=CombinedReward(tuple(rewards), tuple(weights)),
                          action_parser=DiscreteAction(),
                          state_setter=state_setter)


# ---------------------------------------------------------------------------
# CURRICULUM TRACKER
# ---------------------------------------------------------------------------

class CurriculumTracker:
    """
    Tracks training metrics each training iteration and manages:
      1. Stage advancement — entropy-based: advance when entropy drops below 1.0.
         Target operating range is 1.0–2.0 nats. Crossing the floor signals
         policy convergence on the current stage's behavior.
      2. Clip fraction management — reduce policy_lr if updates are too large.
      3. Value loss management — reduce policy_lr if critic is clueless.
      4. Stage fallback — advance after MAX_STAGE_ITERS regardless, to avoid getting stuck.

    When a change is needed, sets restart_requested = True. Caller saves checkpoint
    and calls _restart() to reload everything cleanly.
    """

    REWARD_WINDOW           = 20    # training iters for rolling reward average
    VALUE_LOSS_WINDOW       = 3     # training iters for rolling value loss average
    ADVANCE_THRESHOLDS      = [1.20, 1.12, 1.18, 1.65, 2.0, 2.3]  # stage 0→1 … 5→6
    MAX_STAGE_ITERS         = [60,   80,   100,  100,  120,  150]  # fallback advance (iters)
    VALUE_LOSS_TARGET       = 1.0   # primary advance trigger: value loss drops below this
    #   Value loss tiers:  >100 = clueless  |  60-100 = learning  |  20-60 = working well
    #                        5-20 = strong  |    1-5  = converged  |    <1  = advance stage
    VALUE_LOSS_HIGH_THRESH  = 100.0 # reduce policy_lr when value loss stays above this (clueless tier)
    VALUE_LOSS_HIGH_ITERS   = 5     # consecutive training iters before acting on high vl
    CLIP_HIGH_THRESH        = 0.15  # clip fraction above this is too high
    CLIP_HIGH_ITERS         = 3     # consecutive training iters before acting on clip
    # Entropy floors per stage — when entropy drops below the floor, advance to the next stage.
    # Target operating range is 1.0–2.0 nats. Floor=1.0 is the advance trigger.
    ENTROPY_FLOORS          = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ENTROPY_CEILING         = 2.0   # upper bound of target range (informational)

    def __init__(self, state):
        self.stage                = state["stage"]
        self.policy_lr            = state["policy_lr"]
        self.critic_lr            = state["critic_lr"]
        self.ent_coef             = state["ent_coef"]
        self.reward_window        = list(state.get("reward_window", []))
        self.clip_high_count      = state.get("clip_high_count", 0)
        self.stage_iter_count     = state.get("stage_iter_count", 0)
        self.value_loss_window    = list(state.get("value_loss_window", []))[-self.VALUE_LOSS_WINDOW:]
        self.value_loss_high_count = state.get("value_loss_high_count", 0)
        self.restart_requested    = False
        self.restart_reason       = ""

    def to_state_dict(self):
        return {
            "stage":                self.stage,
            "policy_lr":            self.policy_lr,
            "critic_lr":            self.critic_lr,
            "ent_coef":             self.ent_coef,
            "reward_window":        self.reward_window[-self.REWARD_WINDOW:],
            "clip_high_count":      self.clip_high_count,
            "stage_iter_count":     self.stage_iter_count,
            "value_loss_window":    self.value_loss_window[-self.VALUE_LOSS_WINDOW:],
            "value_loss_high_count": self.value_loss_high_count,
        }

    def update(self, mean_reward, entropy, clip_fraction, value_loss):
        """
        Call after each TRAINING iteration (not accum iters).
        Returns True if a restart is needed.
        """
        self.reward_window.append(mean_reward)
        if len(self.reward_window) > self.REWARD_WINDOW:
            self.reward_window.pop(0)
        self.value_loss_window.append(value_loss)
        if len(self.value_loss_window) > self.VALUE_LOSS_WINDOW:
            self.value_loss_window.pop(0)
        self.stage_iter_count += 1

        vl_avg = sum(self.value_loss_window) / max(len(self.value_loss_window), 1)

        # 1. VALUE LOSS TOO HIGH — reduce policy_lr to let critic catch up.
        #    Only count when window is full (avoids false triggers on post-restart warmup).
        vl_window_full = len(self.value_loss_window) >= self.VALUE_LOSS_WINDOW
        if vl_window_full and vl_avg > self.VALUE_LOSS_HIGH_THRESH:
            self.value_loss_high_count += 1
        else:
            self.value_loss_high_count = 0

        if self.value_loss_high_count >= self.VALUE_LOSS_HIGH_ITERS:
            new_lr = round(self.policy_lr * 0.80, 7)
            if new_lr >= 2e-4:
                self.policy_lr = new_lr
                self.value_loss_high_count = 0
                self.restart_reason = (
                    f"Value loss avg {vl_avg:.1f} > {self.VALUE_LOSS_HIGH_THRESH} "
                    f"for {self.VALUE_LOSS_HIGH_ITERS} iters "
                    f"- policy_lr reduced to {self.policy_lr:.2e}"
                )
                self.restart_requested = True
                return True
            else:
                # LR already at floor (2e-4) — reset count so we don't spam the check
                self.value_loss_high_count = 0

        # 2. CLIP FRACTION — reduce policy_lr if consistently too high
        if clip_fraction > self.CLIP_HIGH_THRESH:
            self.clip_high_count += 1
        else:
            self.clip_high_count = 0

        if self.clip_high_count >= self.CLIP_HIGH_ITERS:
            new_lr = round(self.policy_lr * 0.80, 7)
            if new_lr >= 2e-4:
                self.policy_lr = new_lr
                self.clip_high_count = 0
                self.restart_reason = (
                    f"Clip fraction >{self.CLIP_HIGH_THRESH} for {self.CLIP_HIGH_ITERS} iters "
                    f"- policy_lr reduced to {self.policy_lr:.2e}"
                )
                self.restart_requested = True
                return True

        # 3. ENTROPY-BASED STAGE ADVANCEMENT — advance when entropy drops below floor (1.0).
        #    Target operating range: 1.0–2.0 nats. As the bot converges on the current
        #    stage's behavior its policy entropy compresses; crossing the floor signals
        #    mastery and unlocks the next stage.
        entropy_floor = self.ENTROPY_FLOORS[min(self.stage, len(self.ENTROPY_FLOORS) - 1)]
        if entropy_floor is not None and entropy < entropy_floor:
            if self.stage < len(self.ADVANCE_THRESHOLDS):
                old_stage = self.stage
                old_count = self.stage_iter_count
                self.stage += 1
                self.reward_window.clear()
                self.value_loss_window.clear()
                self.stage_iter_count = 0
                self.value_loss_high_count = 0
                self.clip_high_count = 0
                self.restart_reason = (
                    f"Stage {old_stage} -> {self.stage} "
                    f"({CURRICULUM_STAGE_NAMES[self.stage]}): "
                    f"entropy {entropy:.3f} < floor {entropy_floor:.1f} "
                    f"after {old_count} training iters"
                )
                self.restart_requested = True
                return True

        # 4. STAGE FALLBACK — advance regardless after max iters to avoid getting stuck.
        if self.stage < len(self.ADVANCE_THRESHOLDS):
            max_iters = self.MAX_STAGE_ITERS[self.stage]
            if self.stage_iter_count >= max_iters:
                old_stage = self.stage
                old_count = self.stage_iter_count
                self.stage += 1
                self.reward_window.clear()
                self.value_loss_window.clear()
                self.stage_iter_count = 0
                self.value_loss_high_count = 0
                self.clip_high_count = 0
                self.restart_reason = (
                    f"Stage {old_stage} -> {self.stage} "
                    f"({CURRICULUM_STAGE_NAMES[self.stage]}): "
                    f"fallback after {old_count} training iters"
                )
                self.restart_requested = True
                return True

        return False


# ---------------------------------------------------------------------------
# LEARNER
# ---------------------------------------------------------------------------

class Learner:
    def __init__(self, n_proc=40, ts_per_iteration=100000):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoints_folder = "Checkpoints_2v2"
        self.epoch = 0

        # Load curriculum state — this drives both env rewards (via env_factory)
        # and hyperparameters applied here.
        curriculum_state = load_curriculum_state()
        self.curriculum = CurriculumTracker(curriculum_state)

        self.ts_per_iteration = ts_per_iteration

        stage_name = CURRICULUM_STAGE_NAMES[self.curriculum.stage]
        print(f"[*] Curriculum Stage {self.curriculum.stage}: {stage_name}")
        print(f"[*] Steps per iteration: {ts_per_iteration:,}")
        print(f"[*] Hyperparams — policy_lr={self.curriculum.policy_lr:.2e}  "
              f"critic_lr={self.curriculum.critic_lr:.2e}  ent_coef={self.curriculum.ent_coef}")

        self.agent = rlgym_ppo.batched_agents.BatchedAgentManager(None, min_inference_size=32)
        print(f"[*] Starting Lucifer v4.5 on {n_proc} processes...")
        shapes = self.agent.init_processes(n_processes=n_proc, build_env_fn=env_factory, spawn_delay=0.8)

        self.ppo_learner = rlgym_ppo.ppo.PPOLearner(
            obs_space_size=np.prod(shapes[0]),
            act_space_size=shapes[1],
            device=self.device,
            batch_size=self.ts_per_iteration,
            mini_batch_size=25000,
            n_epochs=3,
            policy_type=shapes[2],
            policy_layer_sizes=(2048, 2048, 1024, 1024),
            critic_layer_sizes=(2048, 2048, 1024, 1024),
            continuous_var_range=0.1,
            policy_lr=self.curriculum.policy_lr,
            critic_lr=self.curriculum.critic_lr,
            clip_range=0.2,
            ent_coef=self.curriculum.ent_coef,
        )
        self.agent.policy = self.ppo_learner.policy

        self.experience_buffer = rlgym_ppo.ppo.ExperienceBuffer(
            max_size=self.ts_per_iteration + 20000, device=self.device, seed=123)
        self._accumulated_steps = 0

        # Per-parameter gradient clipping hooks — hard-clamp to [-1, 1], zero NaN/Inf.
        grad_clip_val = 1.0
        def make_clip_hook(name):
            def clip_hook(grad):
                if grad is not None and not torch.isfinite(grad).all():
                    print(f"[!] NaN/Inf gradient in {name} — zeroing.")
                    return torch.zeros_like(grad)
                return torch.clamp(grad, -grad_clip_val, grad_clip_val)
            return clip_hook

        for name, param in self.ppo_learner.policy.named_parameters():
            param.register_hook(make_clip_hook(f"policy/{name}"))
        for name, param in self.ppo_learner.value_net.named_parameters():
            param.register_hook(make_clip_hook(f"critic/{name}"))
        print("[*] Gradient clipping hooks registered on policy and critic.")

        self.load_latest_checkpoint()

    def _restart(self):
        """Replace this process with a fresh copy (curriculum state already saved by caller)."""
        for method in ('stop', 'terminate', 'shutdown', 'close'):
            fn = getattr(self.agent, method, None)
            if fn:
                try: fn()
                except: pass
                break
        kill_zombies()
        sys.stdout.flush()
        print("[CURRICULUM] RESTARTING...\n")
        os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)

    def learn(self):
        iter_start_time = time.time()
        stage_name = CURRICULUM_STAGE_NAMES[self.curriculum.stage]
        print(f"[*] Iteration {self.epoch + 1} [Stage {self.curriculum.stage}: {stage_name}]: Collecting steps...")

        t0 = time.time()
        exp, metrics, steps_gained, _ = self.agent.collect_timesteps(self.ts_per_iteration)
        t_coll = time.time() - t0

        if steps_gained < 1000: return

        states, actions, log_probs, rewards, next_states, dones, truncated = exp
        val_inp = torch.as_tensor(np.vstack([states, next_states[-1:]]), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            val_preds = self.ppo_learner.value_net(val_inp).cpu().flatten().numpy()

        rewards_arr = np.array(rewards)
        rewards_std = rewards_arr.std()
        if rewards_std > 1e-6:
            rewards_norm = (rewards_arr - rewards_arr.mean()) / (rewards_std + 1e-8)
            rewards_norm = np.clip(rewards_norm, -10.0, 10.0)
        else:
            rewards_norm = rewards_arr - rewards_arr.mean()

        vt, adv, _ = rlgym_util.torch_functions.compute_gae(
            rewards_norm, dones, truncated, np.nan_to_num(val_preds))
        self.experience_buffer.submit_experience(
            states, actions, log_probs, rewards_norm, next_states, dones, truncated, vt, adv)
        self._accumulated_steps += steps_gained

        t1 = time.time()
        if self._accumulated_steps >= self.ts_per_iteration:
            report = self.ppo_learner.learn(self.experience_buffer)
            self.experience_buffer.clear()
            self._accumulated_steps = 0
        else:
            report = {k: 0 for k in ['PPO Batch Consumption Time', 'Cumulative Model Updates',
                                      'Policy Entropy', 'Mean KL Divergence', 'Value Function Loss',
                                      'SB3 Clip Fraction', 'Policy Update Magnitude',
                                      'Value Function Update Magnitude']}
        t_cons = time.time() - t1

        # NaN weight detector
        nan_found = False
        for name, param in self.ppo_learner.policy.named_parameters():
            if not torch.isfinite(param).all():
                print(f"[!!!] NaN/Inf in POLICY weight: {name}")
                nan_found = True
        for name, param in self.ppo_learner.value_net.named_parameters():
            if not torch.isfinite(param).all():
                print(f"[!!!] NaN/Inf in CRITIC weight: {name}")
                nan_found = True
        if nan_found:
            print("[!!!] NaN detected. Network unrecoverable — delete checkpoints and restart.")

        self.epoch += 1
        total_time = time.time() - iter_start_time
        is_training_iter = report.get('PPO Batch Consumption Time', 0) > 0.5

        if is_training_iter:
            entropy      = report.get('Policy Entropy', 0)
            clip         = report.get('SB3 Clip Fraction', 0)
            value_loss   = report.get('Value Function Loss', 0)
            mean_reward  = float(np.mean(rewards))
            vl_window    = self.curriculum.value_loss_window
            vl_avg       = sum(vl_window) / max(len(vl_window), 1) if vl_window else value_loss

            print("\n" + "=" * 45)
            print(f"LUCIFER v4.5 — Stage {self.curriculum.stage} ({stage_name}) — Iter {self.epoch}  [TRAINING]")
            print(f"[-] Total Steps       : {self.agent.cumulative_timesteps:,}")
            print(f"[-] Iter Time         : {total_time:.2f} seconds")
            print(f"[-] Global SPS        : {int(steps_gained/total_time)}")
            print(f"[-] Mean Reward       : {mean_reward:.6f}")
            print(f"[-] Cumul. Updates    : {int(report.get('Cumulative Model Updates', 0))}")
            print(f"[-] Entropy           : {entropy:.4f}  (floor: {self.curriculum.ENTROPY_FLOORS[min(self.curriculum.stage, 6)]})")
            print(f"[-] Mean KL Div       : {report.get('Mean KL Divergence', 0):.6f}")
            vl_tier = ("CLUELESS"    if vl_avg > 100  else
                       "learning"    if vl_avg > 60   else
                       "working"     if vl_avg > 20   else
                       "strong"      if vl_avg > 5    else
                       "converged"   if vl_avg > 1    else
                       "ADVANCE")
            print(f"[-] Value Loss        : {value_loss:.4f}  (avg: {vl_avg:.4f})  [{vl_tier}]")
            print(f"[-] Batch Cons. Time  : {report.get('PPO Batch Consumption Time', 0):.4f}s")
            print(f"[-] Clip Fraction     : {clip:.4f}")
            print(f"[-] Policy Upd. Mag.  : {report.get('Policy Update Magnitude', 0):.6f}")
            print(f"[-] Stage Iter Count  : {self.curriculum.stage_iter_count}")
            print("=" * 45)

            # Feed metrics to curriculum tracker
            if self.curriculum.update(mean_reward, entropy, clip, value_loss):
                # Save state FIRST — if anything below crashes, the stage advance is not lost.
                save_curriculum_state(self.curriculum.to_state_dict())
                print(f"\n[CURRICULUM] {self.curriculum.restart_reason}")
                print(f"[CURRICULUM] Saving checkpoint before restart...")
                self.save_checkpoint(f"lucifer_curriculum_s{self.curriculum.stage}_epoch_{self.epoch}")
                self._restart()

        else:
            print(f"  [ACCUM] iter {self.epoch:>4} | steps {self.agent.cumulative_timesteps:>12,} | "
                  f"reward {np.mean(rewards):.4f} | stage {self.curriculum.stage}")

    def save_checkpoint(self, label):
        path = os.path.join(self.checkpoints_folder, label)
        os.makedirs(path, exist_ok=True)
        self.ppo_learner.save_to(path)
        with open(os.path.join(path, "VARS.json"), "w") as f:
            json.dump({"steps": self.agent.cumulative_timesteps, "epoch": self.epoch}, f)

        import shutil
        all_ckpts = sorted([
            d for d in os.listdir(self.checkpoints_folder)
            if os.path.isdir(os.path.join(self.checkpoints_folder, d))
        ], key=lambda d: os.path.getmtime(os.path.join(self.checkpoints_folder, d)))
        while len(all_ckpts) > 3:
            oldest = all_ckpts.pop(0)
            oldest_path = os.path.join(self.checkpoints_folder, oldest)
            shutil.rmtree(oldest_path)
            print(f"[*] Deleted old checkpoint: {oldest}")

    def load_latest_checkpoint(self):
        if not os.path.exists(self.checkpoints_folder): return
        potential = [d for d in os.listdir(self.checkpoints_folder)
                     if os.path.isdir(os.path.join(self.checkpoints_folder, d))]
        if potential:
            latest = sorted(potential, key=lambda d: os.path.getmtime(
                os.path.join(self.checkpoints_folder, d)))[-1]
            path = os.path.join(self.checkpoints_folder, latest)
            if os.path.exists(os.path.join(path, "VARS.json")):
                self.ppo_learner.load_from(path)
                with open(os.path.join(path, "VARS.json"), "r") as f:
                    v = json.load(f)
                    self.agent.cumulative_timesteps, self.epoch = v["steps"], v["epoch"]
                print(f"[*] Loaded checkpoint at epoch {self.epoch} ({self.agent.cumulative_timesteps:,} steps)")


if __name__ == "__main__":
    try:
        learner = Learner(n_proc=40)
        while True:
            learner.learn()
            if learner.epoch % 10 == 0:
                learner.save_checkpoint(f"lucifer_epoch_{learner.epoch}")
    except KeyboardInterrupt:
        _devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull_fd, 2)
        os.close(_devnull_fd)
        print("\n[!] MANUAL STOP.")
        if 'learner' in locals():
            label = f"lucifer_manual_epoch_{learner.epoch}"
            print(f"[*] Saving checkpoint: {label}")
            learner.save_checkpoint(label)
            save_curriculum_state(learner.curriculum.to_state_dict())
            print(f"[*] Checkpoint + curriculum state saved.")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        if 'learner' in locals():
            for method in ('stop', 'terminate', 'shutdown', 'close'):
                fn = getattr(learner.agent, method, None)
                if fn:
                    try: fn()
                    except: pass
                    break
        kill_zombies()
        sys.stdout.flush()
        print("[*] BYE.\n")
        os._exit(0)
