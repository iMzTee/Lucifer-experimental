"""vectorized_env.py — Vectorized reward, obs, and terminal computation.

Replaces per-env Python loops with batch numpy operations across all envs/agents.
All functions operate on (E, A, ...) arrays where E=n_envs, A=n_agents_per_env.

v2.0 — Complete reward rewrite: 12 signals, team spirit, 4-stage curriculum.
"""

import numpy as np

try:
    import fast_extract
    _HAS_FAST_EXTRACT = True
except ImportError:
    _HAS_FAST_EXTRACT = False

# ── Constants (from rlgym_sim.utils.common_values) ──
BLUE_TEAM = 0
ORANGE_TEAM = 1
BALL_RADIUS = 92.75
BALL_MAX_SPEED = 6000
CAR_MAX_SPEED = 2300
BACK_NET_Y = 6000
BACK_WALL_Y = 5120
GOAL_HEIGHT = 642.775
ORANGE_GOAL_BACK = np.array([0.0, BACK_NET_Y, GOAL_HEIGHT / 2], dtype=np.float32)
BLUE_GOAL_BACK = np.array([0.0, -BACK_NET_Y, GOAL_HEIGHT / 2], dtype=np.float32)
POS_COEF = np.float32(1 / 2300)
LIN_VEL_COEF = np.float32(1 / 2300)
ANG_VEL_COEF = np.float32(1 / np.pi)
JUMP_TIMER_SECONDS = 1.25

LARGE_BOOST_PAD_POS = np.array([
    [-3584.0, 0.0, 73.0], [3584.0, 0.0, 73.0],
    [-3072.0, 4096.0, 73.0], [3072.0, 4096.0, 73.0],
    [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0],
], dtype=np.float32)

# 2v2 layout: players sorted by car_id → [B0, B1, O0, O1]
ALLY_IDX = np.array([1, 0, 3, 2])
ENEMY0_IDX = np.array([2, 2, 0, 0])
ENEMY1_IDX = np.array([3, 3, 1, 1])
IS_ORANGE = np.array([False, False, True, True])


# ─────────────────────────────────────────────────────────
# STAGE CONFIGURATION — 4 stages (down from 7)
# ─────────────────────────────────────────────────────────

STAGE_CONFIG = {
    0: {"tick_skip": 8, "timeout": 300},
    1: {"tick_skip": 8, "timeout": 400},
    2: {"tick_skip": 4, "timeout": 600},
    3: {"tick_skip": 2, "timeout": 1200},
}

# Continuous reward weights: 13 signals [R1..R13]
# [R1:VelBallGoal, R2:BallGoalDist, R3:TouchQuality, R4:PlayerBallProxVel,
#  R5:Kickoff, R6:DefensivePos, R7:BoostEff, R8:DemoAttempt,
#  R9:AirControl, R10:FlipReset, R11:AngVel, R12:Speed, R13:BallAccel]
STAGE_WEIGHTS = {
    0: np.array([2.0, 1.0, 2.0, 2.0, 3.0, 0.5, 0.5, 0.0, 0.5, 0.0,  0.0,   1.0, 1.5], dtype=np.float32),
    1: np.array([5.0, 3.0, 3.0, 1.5, 2.0, 1.5, 1.0, 0.5, 1.5, 0.0,  0.005, 0.5, 2.0], dtype=np.float32),
    2: np.array([5.0, 3.0, 3.0, 1.0, 2.0, 1.5, 1.0, 1.0, 2.0, 5.0,  0.005, 0.3, 2.0], dtype=np.float32),
    3: np.array([5.0, 3.0, 3.0, 0.5, 2.0, 1.5, 1.0, 1.0, 2.0, 10.0, 0.005, 0.2, 2.0], dtype=np.float32),
}

# Event weights: [goal, team_score_inc, opp_score_inc(concede), touch, shot, save, demo, boost_pickup]
# Same across all stages — event magnitudes provide the scaling.
EVENT_WEIGHTS = {
    0: np.array([10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0], dtype=np.float32),
    1: np.array([10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0], dtype=np.float32),
    2: np.array([10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0], dtype=np.float32),
    3: np.array([10.0, 0.0, -7.0, 0.5, 3.0, 5.0, 8.0, 0.0], dtype=np.float32),
}

# Team spirit blending: reward = (1-ts) * individual + ts * team_mean
TEAM_SPIRIT = {0: 0.0, 1: 0.3, 2: 0.5, 3: 0.6}


def batch_quat_to_rot_mtx(quats):
    """Convert (N, 4) quaternions to (N, 3, 3) rotation matrices (rlgym_sim convention)."""
    w = -quats[:, 0]
    x = -quats[:, 1]
    y = -quats[:, 2]
    z = -quats[:, 3]
    norm = np.einsum('ij,ij->i', quats, quats)
    s = np.where(norm != 0, 1.0 / norm, 0.0)
    rot = np.zeros((len(quats), 3, 3), dtype=np.float32)
    rot[:, 0, 0] = 1.0 - 2.0 * s * (y*y + z*z)
    rot[:, 1, 0] = 2.0 * s * (x*y + z*w)
    rot[:, 2, 0] = 2.0 * s * (x*z - y*w)
    rot[:, 0, 1] = 2.0 * s * (x*y - z*w)
    rot[:, 1, 1] = 1.0 - 2.0 * s * (x*x + z*z)
    rot[:, 2, 1] = 2.0 * s * (y*z + x*w)
    rot[:, 0, 2] = 2.0 * s * (x*z + y*w)
    rot[:, 1, 2] = 2.0 * s * (y*z - x*w)
    rot[:, 2, 2] = 1.0 - 2.0 * s * (x*x + y*y)
    return rot


class BatchState:
    """Pre-allocated numpy arrays for vectorized state data across all envs."""

    def __init__(self, n_envs, n_agents=4):
        E, A = n_envs, n_agents
        self.n_envs = E
        self.n_agents = A

        # Ball
        self.ball_pos = np.zeros((E, 3), dtype=np.float32)
        self.ball_vel = np.zeros((E, 3), dtype=np.float32)
        self.ball_ang = np.zeros((E, 3), dtype=np.float32)
        self.inv_ball_pos = np.zeros((E, 3), dtype=np.float32)
        self.inv_ball_vel = np.zeros((E, 3), dtype=np.float32)
        self.inv_ball_ang = np.zeros((E, 3), dtype=np.float32)

        # Boost pads
        self.boost_pads = np.zeros((E, 34), dtype=np.float32)
        self.inv_boost_pads = np.zeros((E, 34), dtype=np.float32)

        # Scores
        self.blue_score = np.zeros(E, dtype=np.int32)
        self.orange_score = np.zeros(E, dtype=np.int32)

        # Player core (normal + inverted)
        self.player_pos = np.zeros((E, A, 3), dtype=np.float32)
        self.player_vel = np.zeros((E, A, 3), dtype=np.float32)
        self.player_ang_vel = np.zeros((E, A, 3), dtype=np.float32)
        self.inv_player_pos = np.zeros((E, A, 3), dtype=np.float32)
        self.inv_player_vel = np.zeros((E, A, 3), dtype=np.float32)
        self.inv_player_ang_vel = np.zeros((E, A, 3), dtype=np.float32)

        # Derived rotation vectors (read directly from raw data, no quat conversion)
        self.player_fwd = np.zeros((E, A, 3), dtype=np.float32)
        self.player_up = np.zeros((E, A, 3), dtype=np.float32)
        self.inv_player_fwd = np.zeros((E, A, 3), dtype=np.float32)
        self.inv_player_up = np.zeros((E, A, 3), dtype=np.float32)

        # Player attributes
        self.player_boost = np.zeros((E, A), dtype=np.float32)
        self.player_on_ground = np.zeros((E, A), dtype=np.float32)
        self.player_has_flip = np.zeros((E, A), dtype=np.float32)
        self.player_is_demoed = np.zeros((E, A), dtype=np.float32)
        self.player_ball_touched = np.zeros((E, A), dtype=np.float32)
        self.player_team = np.zeros((E, A), dtype=np.int32)
        self.player_car_id = np.zeros((E, A), dtype=np.int32)

        # Event tracking (for EventReward)
        self.player_goals = np.zeros((E, A), dtype=np.float32)
        self.player_saves = np.zeros((E, A), dtype=np.float32)
        self.player_shots = np.zeros((E, A), dtype=np.float32)
        self.player_demos = np.zeros((E, A), dtype=np.float32)

    def extract(self, game_states):
        """Extract data from list of GameState objects into batch arrays.
        Used only for init/reset (not hot path)."""
        for i, state in enumerate(game_states):
            self.ball_pos[i] = state.ball.position
            self.ball_vel[i] = state.ball.linear_velocity
            self.ball_ang[i] = state.ball.angular_velocity
            self.inv_ball_pos[i] = state.inverted_ball.position
            self.inv_ball_vel[i] = state.inverted_ball.linear_velocity
            self.inv_ball_ang[i] = state.inverted_ball.angular_velocity
            self.boost_pads[i] = state.boost_pads
            self.inv_boost_pads[i] = state.inverted_boost_pads
            self.blue_score[i] = state.blue_score
            self.orange_score[i] = state.orange_score
            for j, p in enumerate(state.players):
                self.player_pos[i, j] = p.car_data.position
                self.player_vel[i, j] = p.car_data.linear_velocity
                self.player_ang_vel[i, j] = p.car_data.angular_velocity
                self.inv_player_pos[i, j] = p.inverted_car_data.position
                self.inv_player_vel[i, j] = p.inverted_car_data.linear_velocity
                self.inv_player_ang_vel[i, j] = p.inverted_car_data.angular_velocity
                self.player_boost[i, j] = p.boost_amount
                self.player_on_ground[i, j] = float(p.on_ground)
                self.player_has_flip[i, j] = float(p.has_flip)
                self.player_is_demoed[i, j] = float(p.is_demoed)
                self.player_ball_touched[i, j] = float(p.ball_touched)
                self.player_team[i, j] = p.team_num
                self.player_car_id[i, j] = p.car_id
                self.player_goals[i, j] = p.match_goals
                self.player_saves[i, j] = p.match_saves
                self.player_shots[i, j] = p.match_shots
                self.player_demos[i, j] = p.match_demolishes
                # Read rotation vectors directly from car_data
                self.player_fwd[i, j] = p.car_data._rotation_mtx[:, 0]
                self.player_up[i, j] = p.car_data._rotation_mtx[:, 2]
                self.inv_player_fwd[i, j] = p.inverted_car_data._rotation_mtx[:, 0]
                self.inv_player_up[i, j] = p.inverted_car_data._rotation_mtx[:, 2]

    def extract_single(self, i, state):
        """Extract data for a single env index from one GameState.
        Used only for reset (not hot path)."""
        self.ball_pos[i] = state.ball.position
        self.ball_vel[i] = state.ball.linear_velocity
        self.ball_ang[i] = state.ball.angular_velocity
        self.inv_ball_pos[i] = state.inverted_ball.position
        self.inv_ball_vel[i] = state.inverted_ball.linear_velocity
        self.inv_ball_ang[i] = state.inverted_ball.angular_velocity
        self.boost_pads[i] = state.boost_pads
        self.inv_boost_pads[i] = state.inverted_boost_pads
        self.blue_score[i] = state.blue_score
        self.orange_score[i] = state.orange_score
        for j, p in enumerate(state.players):
            self.player_pos[i, j] = p.car_data.position
            self.player_vel[i, j] = p.car_data.linear_velocity
            self.player_ang_vel[i, j] = p.car_data.angular_velocity
            self.inv_player_pos[i, j] = p.inverted_car_data.position
            self.inv_player_vel[i, j] = p.inverted_car_data.linear_velocity
            self.inv_player_ang_vel[i, j] = p.inverted_car_data.angular_velocity
            self.player_boost[i, j] = p.boost_amount
            self.player_on_ground[i, j] = float(p.on_ground)
            self.player_has_flip[i, j] = float(p.has_flip)
            self.player_is_demoed[i, j] = float(p.is_demoed)
            self.player_ball_touched[i, j] = float(p.ball_touched)
            self.player_team[i, j] = p.team_num
            self.player_car_id[i, j] = p.car_id
            self.player_goals[i, j] = p.match_goals
            self.player_saves[i, j] = p.match_saves
            self.player_shots[i, j] = p.match_shots
            self.player_demos[i, j] = p.match_demolishes
            self.player_fwd[i, j] = p.car_data._rotation_mtx[:, 0]
            self.player_up[i, j] = p.car_data._rotation_mtx[:, 2]
            self.inv_player_fwd[i, j] = p.inverted_car_data._rotation_mtx[:, 0]
            self.inv_player_up[i, j] = p.inverted_car_data._rotation_mtx[:, 2]

    def extract_raw(self, raw_states, car_orders, car_objects_list):
        """Extract directly from raw arena.get_gym_state() tuples.

        Bypasses GameState/PlayerData/PhysicsObject construction entirely.
        Reads rotation vectors directly from raw data (no quat conversion needed).

        Uses C++ extension (fast_extract) when available for ~5x speedup on the
        data scatter loop. Falls back to Python if not built.

        Args:
            raw_states: list of get_gym_state() tuples, one per env
            car_orders: list of np.array mapping arena-order → ordered player index
            car_objects_list: list of lists of car objects in ordered order (for has_flip)
        """
        E = len(raw_states)
        P = len(car_orders[0])

        if _HAS_FAST_EXTRACT:
            # ── C++ fast path: pre-stack into contiguous arrays, then C++ scatter ──
            pad_norm  = np.array([raw[1][0] for raw in raw_states], dtype=np.float32)
            pad_inv   = np.array([raw[1][1] for raw in raw_states], dtype=np.float32)
            ball_norm = np.array([raw[2][0] for raw in raw_states], dtype=np.float32)
            ball_inv  = np.array([raw[2][1] for raw in raw_states], dtype=np.float32)
            car_n = np.array(
                [[raw[3 + k][0] for k in range(P)] for raw in raw_states],
                dtype=np.float32)
            car_i = np.array(
                [[raw[3 + k][1] for k in range(P)] for raw in raw_states],
                dtype=np.float32)
            orders_arr = np.ascontiguousarray(np.array(car_orders, dtype=np.int32))
            scores_arr = np.array(
                [[int(raw[0][2]), int(raw[0][3])] for raw in raw_states],
                dtype=np.int32)

            fast_extract.extract_raw_fast(
                pad_norm, pad_inv, ball_norm, ball_inv,
                car_n, car_i, orders_arr, scores_arr, self)

            # has_flip with grounded-player optimization:
            # skip car.get_state() for on_ground (always has_flip) and demoed (never has_flip)
            for i in range(E):
                for j, car in enumerate(car_objects_list[i]):
                    if self.player_on_ground[i, j] == 1.0:
                        self.player_has_flip[i, j] = 1.0
                    elif self.player_is_demoed[i, j] == 1.0:
                        self.player_has_flip[i, j] = 0.0
                    else:
                        cs = car.get_state()
                        self.player_has_flip[i, j] = float(
                            cs.air_time_since_jump < JUMP_TIMER_SECONDS
                            and not (cs.has_flipped or cs.has_double_jumped))
            return

        # ── Python fallback ──
        for i, (raw, order, cars) in enumerate(zip(raw_states, car_orders, car_objects_list)):
            game_data = raw[0]
            pad_data = raw[1]
            ball_data = raw[2]

            self.blue_score[i] = int(game_data[2])
            self.orange_score[i] = int(game_data[3])

            bd = ball_data[0]
            self.ball_pos[i] = bd[:3]
            self.ball_vel[i] = bd[7:10]
            self.ball_ang[i] = bd[10:13]
            bi = ball_data[1]
            self.inv_ball_pos[i] = bi[:3]
            self.inv_ball_vel[i] = bi[7:10]
            self.inv_ball_ang[i] = bi[10:13]

            self.boost_pads[i] = pad_data[0]
            self.inv_boost_pads[i] = pad_data[1]

            for k in range(len(order)):
                j = order[k]
                d = raw[3 + k][0]
                di = raw[3 + k][1]

                self.player_team[i, j] = int(d[1])
                self.player_goals[i, j] = d[2]
                self.player_saves[i, j] = d[3]
                self.player_shots[i, j] = d[4]
                self.player_demos[i, j] = d[5]
                self.player_is_demoed[i, j] = d[7]
                self.player_on_ground[i, j] = d[8]
                self.player_ball_touched[i, j] = d[9]
                self.player_boost[i, j] = d[10] / 100.0
                self.player_pos[i, j] = d[11:14]
                self.player_vel[i, j] = d[18:21]
                self.player_ang_vel[i, j] = d[21:24]
                self.player_fwd[i, j] = d[24:27]
                self.player_up[i, j] = d[30:33]
                self.player_car_id[i, j] = int(d[0])

                self.inv_player_pos[i, j] = di[11:14]
                self.inv_player_vel[i, j] = di[18:21]
                self.inv_player_ang_vel[i, j] = di[21:24]
                self.inv_player_fwd[i, j] = di[24:27]
                self.inv_player_up[i, j] = di[30:33]

            # has_flip with grounded-player optimization
            for j, car in enumerate(cars):
                if self.player_on_ground[i, j] == 1.0:
                    self.player_has_flip[i, j] = 1.0
                elif self.player_is_demoed[i, j] == 1.0:
                    self.player_has_flip[i, j] = 0.0
                else:
                    cs = car.get_state()
                    self.player_has_flip[i, j] = float(
                        cs.air_time_since_jump < JUMP_TIMER_SECONDS
                        and not (cs.has_flipped or cs.has_double_jumped))

    def extract_raw_single(self, i, raw, order, cars):
        """Extract raw data for a single env. Used for post-reset extraction."""
        game_data = raw[0]
        pad_data = raw[1]
        ball_data = raw[2]

        self.blue_score[i] = int(game_data[2])
        self.orange_score[i] = int(game_data[3])

        bd = ball_data[0]
        self.ball_pos[i] = bd[:3]
        self.ball_vel[i] = bd[7:10]
        self.ball_ang[i] = bd[10:13]
        bi = ball_data[1]
        self.inv_ball_pos[i] = bi[:3]
        self.inv_ball_vel[i] = bi[7:10]
        self.inv_ball_ang[i] = bi[10:13]

        self.boost_pads[i] = pad_data[0]
        self.inv_boost_pads[i] = pad_data[1]

        for k in range(len(order)):
            j = order[k]
            d = raw[3 + k][0]
            di = raw[3 + k][1]

            self.player_team[i, j] = int(d[1])
            self.player_goals[i, j] = d[2]
            self.player_saves[i, j] = d[3]
            self.player_shots[i, j] = d[4]
            self.player_demos[i, j] = d[5]
            self.player_is_demoed[i, j] = d[7]
            self.player_on_ground[i, j] = d[8]
            self.player_ball_touched[i, j] = d[9]
            self.player_boost[i, j] = d[10] / 100.0
            self.player_pos[i, j] = d[11:14]
            self.player_vel[i, j] = d[18:21]
            self.player_ang_vel[i, j] = d[21:24]
            self.player_fwd[i, j] = d[24:27]
            self.player_up[i, j] = d[30:33]
            self.player_car_id[i, j] = int(d[0])

            self.inv_player_pos[i, j] = di[11:14]
            self.inv_player_vel[i, j] = di[18:21]
            self.inv_player_ang_vel[i, j] = di[21:24]
            self.inv_player_fwd[i, j] = di[24:27]
            self.inv_player_up[i, j] = di[30:33]

        # has_flip with grounded-player optimization
        for j, car in enumerate(cars):
            if self.player_on_ground[i, j] == 1.0:
                self.player_has_flip[i, j] = 1.0
            elif self.player_is_demoed[i, j] == 1.0:
                self.player_has_flip[i, j] = 0.0
            else:
                cs = car.get_state()
                self.player_has_flip[i, j] = float(
                    cs.air_time_since_jump < JUMP_TIMER_SECONDS
                    and not (cs.has_flipped or cs.has_double_jumped))


# ─────────────────────────────────────────────────────────
# VECTORIZED REWARD COMPUTATION — 12 signals + events + team spirit
# ─────────────────────────────────────────────────────────

class VectorizedRewards:
    """Computes all 12 reward signals for all envs/agents in vectorized numpy.

    Reward signals:
        R1:  VelocityBallToGoal       — dot(ball_dir_to_goal, ball_vel/max)
        R2:  BallGoalDistancePotential — exp(-to_opp/6000) - exp(-to_own/6000)
        R3:  TouchQuality              — height × speed × wall_factor on touch
        R4:  PlayerBallProximityVel    — speed toward ball (closest on team only)
        R5:  KickoffReward             — rush ball during kickoff
        R6:  DefensivePositioning      — support role alignment to own goal
        R7:  BoostEfficiency           — boost pickup reward (small pads 2×)
        R8:  DemoAttempt               — approach speed toward nearest opponent
        R9:  AirControl                — max(dribble, aerial facing)
        R10: FlipResetDetector         — upside-down aerial touch = 10.0
        R11: AngularVelocity           — norm(ang_vel) / 6π
        R12: Speed + Anti-Passive       — norm(vel) / max_speed - 0.1 if idle on ground
        R13: BallAcceleration           — ball speed increase on touch (hit ball hard)
        Events: goal=10, concede=-7, touch=0.5, shot=3, save=5, demo=8
    """

    def __init__(self, n_envs, n_agents=4):
        E, A = n_envs, n_agents
        self._E = E
        self._A = A
        # EventReward state: [goals, team_score, opp_score, touched, shots, saves, demos, boost]
        self.event_last = np.zeros((E, A, 8), dtype=np.float32)
        # KickoffReward state
        self.is_kickoff = np.zeros(E, dtype=bool)
        # BoostEfficiency tracking
        self.prev_player_boost = np.zeros((E, A), dtype=np.float32)
        # BallAcceleration tracking
        self.prev_ball_speed = np.zeros(E, dtype=np.float32)

        # Pre-allocated scratch arrays (reduces GC pressure)
        self._to_ball = np.empty((E, A, 3), dtype=np.float32)
        self._to_ball_dist_sq = np.empty((E, A), dtype=np.float32)
        self._to_ball_dist = np.empty((E, A), dtype=np.float32)
        self._player_speed = np.empty((E, A), dtype=np.float32)
        self._rewards = np.zeros((E, A), dtype=np.float32)
        self._current_ev = np.zeros((E, A, 8), dtype=np.float32)
        self._ball_speed_1d = np.empty(E, dtype=np.float32)
        self._speed_sq = np.empty((E, A), dtype=np.float32)

    def reset_env(self, env_idx, bs):
        """Reset state for a single env after episode end."""
        # EventReward: snapshot current values
        for j in range(bs.n_agents):
            team = bs.player_team[env_idx, j]
            if team == BLUE_TEAM:
                team_score = bs.blue_score[env_idx]
                opp_score = bs.orange_score[env_idx]
            else:
                team_score = bs.orange_score[env_idx]
                opp_score = bs.blue_score[env_idx]
            self.event_last[env_idx, j] = [
                bs.player_goals[env_idx, j], team_score, opp_score,
                bs.player_ball_touched[env_idx, j],
                bs.player_shots[env_idx, j], bs.player_saves[env_idx, j],
                bs.player_demos[env_idx, j], bs.player_boost[env_idx, j]
            ]
        # KickoffReward: check if ball at center
        bp = bs.ball_pos[env_idx]
        self.is_kickoff[env_idx] = (abs(bp[0]) < 50 and abs(bp[1]) < 50 and bp[2] < 120)
        # BoostEfficiency: snapshot current boost
        self.prev_player_boost[env_idx] = bs.player_boost[env_idx]
        # BallAcceleration: snapshot current ball speed
        bv = bs.ball_vel[env_idx]
        self.prev_ball_speed[env_idx] = np.sqrt(bv[0]*bv[0] + bv[1]*bv[1] + bv[2]*bv[2])

    def compute(self, bs, stage):
        """Compute combined rewards for all envs/agents.

        All 12 signals are always computed; stage only changes weights.
        Zero-weight signals are skipped for efficiency.
        Team spirit blends individual + team_mean reward.

        Returns: (E, A) reward array
        """
        E, A = bs.n_envs, bs.n_agents
        weights = STAGE_WEIGHTS.get(stage, STAGE_WEIGHTS[3])
        ev_weights = EVENT_WEIGHTS.get(stage, EVENT_WEIGHTS[3])
        ts = TEAM_SPIRIT.get(stage, 0.6)

        # ── Precompute common values (using pre-allocated arrays) ──
        ball_e = bs.ball_pos[:, None, :]                          # (E, 1, 3)
        np.subtract(ball_e, bs.player_pos, out=self._to_ball)     # (E, A, 3)
        to_ball = self._to_ball
        to_ball_sq = to_ball * to_ball                            # temp alloc for sq
        np.sum(to_ball_sq, axis=2, out=self._to_ball_dist_sq)
        del to_ball_sq
        to_ball_dist_sq = self._to_ball_dist_sq
        np.sqrt(to_ball_dist_sq, out=self._to_ball_dist)
        to_ball_dist = self._to_ball_dist
        to_ball_dir = to_ball / (to_ball_dist[:, :, None] + 1e-6)
        vel_sq = bs.player_vel * bs.player_vel
        np.sum(vel_sq, axis=2, out=self._speed_sq)
        del vel_sq
        np.sqrt(self._speed_sq, out=self._player_speed)
        player_speed = self._player_speed
        is_blue = (bs.player_team == BLUE_TEAM)                   # (E, A)

        # Goal positions per agent (opp = score target, own = defend target)
        opp_goal = np.where(is_blue[:, :, None], ORANGE_GOAL_BACK, BLUE_GOAL_BACK)
        own_goal = np.where(is_blue[:, :, None], BLUE_GOAL_BACK, ORANGE_GOAL_BACK)

        # Closest-on-team mask (for R4 and R6)
        ally_dist = to_ball_dist[:, ALLY_IDX]
        is_closest = to_ball_dist <= ally_dist                    # (E, A)

        # Ball speed — shared between R3 and R5
        np.sum(bs.ball_vel * bs.ball_vel, axis=1, out=self._ball_speed_1d)
        np.sqrt(self._ball_speed_1d, out=self._ball_speed_1d)
        ball_speed_1d = self._ball_speed_1d

        # Initialize combined reward (zero in-place)
        self._rewards[:] = 0.0
        rewards = self._rewards

        # ── R1: VelocityBallToGoal ──
        if weights[0] > 0:
            pos_diff = opp_goal - ball_e
            norm_pd = pos_diff / (np.sqrt(np.sum(pos_diff * pos_diff, axis=2, keepdims=True)) + 1e-6)
            norm_bv = bs.ball_vel[:, None, :] / BALL_MAX_SPEED
            rewards += weights[0] * np.sum(norm_pd * norm_bv, axis=2)

        # ── R2: BallGoalDistancePotential ──
        if weights[1] > 0:
            ball_to_opp = np.sqrt(np.sum((ball_e - opp_goal) ** 2, axis=2))
            ball_to_own = np.sqrt(np.sum((ball_e - own_goal) ** 2, axis=2))
            rewards += weights[1] * (np.exp(-ball_to_opp / 6000.0) - np.exp(-ball_to_own / 6000.0))

        # ── R3: TouchQuality ──
        if weights[2] > 0:
            touched = bs.player_ball_touched
            ball_z = bs.ball_pos[:, 2][:, None]
            ball_speed_e = ball_speed_1d[:, None]
            height_term = 1.0 + np.cbrt(np.maximum(0.0, ball_z - 150.0) / 2044.0) * 2.0
            speed_term = 0.5 + 0.5 * np.clip(ball_speed_e / 2300.0, 0.0, 2.0)
            wall_x = (np.abs(bs.ball_pos[:, 0]) > 3800.0)[:, None]
            wall_y = (np.abs(bs.ball_pos[:, 1]) > 4800.0)[:, None]
            wall_factor = np.where(wall_x | wall_y, 1.5, 1.0)
            rewards += weights[2] * (touched * height_term * speed_term * wall_factor)

        # ── R4: PlayerBallProximityVelocity (closest on team only) ──
        if weights[3] > 0:
            speed_toward_ball = np.maximum(0.0,
                np.sum(bs.player_vel * to_ball_dir, axis=2) / CAR_MAX_SPEED)
            rewards += weights[3] * (speed_toward_ball * is_closest)

        # ── R5: KickoffReward ──
        if weights[4] > 0:
            self.is_kickoff = self.is_kickoff & (ball_speed_1d < 100)
            kick_speed = np.sum(bs.player_vel * to_ball_dir, axis=2)
            rewards += weights[4] * ((np.maximum(0.0, kick_speed / 2300.0) +
                  np.exp(-to_ball_dist / 800.0)) * self.is_kickoff[:, None])
        else:
            self.is_kickoff = self.is_kickoff & (ball_speed_1d < 100)

        # ── R6: DefensivePositioning (support role only) ──
        if weights[5] > 0:
            is_support = ~is_closest
            own_goal_y = np.where(is_blue, -5120.0, 5120.0)
            own_goal_3d = np.zeros((E, A, 3), dtype=np.float32)
            own_goal_3d[:, :, 1] = own_goal_y
            g2b = ball_e - own_goal_3d
            g2p = bs.player_pos - own_goal_3d
            g2b_n = np.sqrt(np.sum(g2b * g2b, axis=2, keepdims=True)) + 1e-6
            g2p_n = np.sqrt(np.sum(g2p * g2p, axis=2, keepdims=True)) + 1e-6
            align = np.maximum(0.0,
                np.sum(g2p * g2b, axis=2) / (g2b_n[:, :, 0] * g2p_n[:, :, 0]))
            dist_ratio = g2p_n[:, :, 0] / g2b_n[:, :, 0]
            gaussian = np.exp(-((dist_ratio - 0.7) ** 2) / (2.0 * 0.15 ** 2))
            rewards += weights[5] * (is_support * align * gaussian)

        # ── R7: BoostEfficiency ──
        if weights[6] > 0:
            boost_gained = np.maximum(0.0, bs.player_boost - self.prev_player_boost)
            is_small = (boost_gained > 0.01) & (boost_gained <= 0.15)
            pad_mult = np.where(is_small, 2.0, 1.0)
            rewards += weights[6] * np.clip(np.sqrt(boost_gained) * pad_mult, 0.0, 0.5)

        # ── R8: DemoAttempt (skip when weight=0 e.g. Stage 0) ──
        if weights[7] > 0:
            opp0_pos = bs.player_pos[:, ENEMY0_IDX]
            opp1_pos = bs.player_pos[:, ENEMY1_IDX]
            to_opp0 = opp0_pos - bs.player_pos
            to_opp1 = opp1_pos - bs.player_pos
            d0 = np.sqrt(np.sum(to_opp0 * to_opp0, axis=2))
            d1 = np.sqrt(np.sum(to_opp1 * to_opp1, axis=2))
            nearest_dist = np.minimum(d0, d1)
            nearest_vec = np.where((d0 <= d1)[:, :, None], to_opp0, to_opp1)
            nearest_dir = nearest_vec / (nearest_dist[:, :, None] + 1e-6)
            speed_to_opp = np.sum(bs.player_vel * nearest_dir, axis=2)
            rewards += weights[7] * (np.exp(-nearest_dist / 500.0) *
                  np.maximum(0.0, speed_to_opp / 2300.0) *
                  (player_speed > 1500.0))

        # ── R9: AirControl = max(dribble, aerial) ──
        if weights[8] > 0:
            bz = bs.ball_pos[:, 2][:, None]
            cz = bs.player_pos[:, :, 2]
            ball_above = bz > (cz + 60.0)
            xy_diff = bs.ball_pos[:, None, :2] - bs.player_pos[:, :, :2]
            xy_dist = np.sqrt(np.sum(xy_diff * xy_diff, axis=2))
            close_overhead = xy_dist < 180.0
            prox = np.clip(1.0 - xy_dist / 180.0, 0.0, 1.0)
            ht = np.clip((bz - cz - 60.0) / 250.0, 0.0, 1.0)
            dribble = np.where(
                ball_above & close_overhead & (bs.player_on_ground > 0.5),
                prox * (0.3 + 0.7 * ht), 0.0)
            off_ground = bs.player_on_ground < 0.5
            facing_ball = np.maximum(0.0, np.sum(bs.player_fwd * to_ball_dir, axis=2))
            aerial = off_ground * facing_ball
            rewards += weights[8] * np.maximum(dribble, aerial)

        # ── R10: FlipResetDetector (skip when weight=0 e.g. Stage 0-1) ──
        if weights[9] > 0:
            car_up_z = bs.player_up[:, :, 2]
            rewards += weights[9] * (bs.player_ball_touched *
                   (bs.player_on_ground < 0.5) *
                   (car_up_z < -0.5) * 10.0)

        # ── R11: AngularVelocity (skip when weight=0 e.g. Stage 0) ──
        if weights[10] > 0:
            rewards += weights[10] * (np.sqrt(np.sum(
                bs.player_ang_vel * bs.player_ang_vel, axis=2)) / (6.0 * np.pi))

        # ── R12: Speed + Anti-Passive Penalty (smooth ramp) ──
        if weights[11] > 0:
            speed_ratio = player_speed / CAR_MAX_SPEED
            # Smooth penalty: linearly ramps from -0.03 (stationary) to 0 (at 600 uu/s)
            # No hard threshold — avoids optimizer discontinuity
            passive_pen = np.maximum(0.0, (1.0 - player_speed / 600.0)) * np.float32(0.03) * (bs.player_is_demoed < 0.5) * (bs.player_on_ground > 0.5)
            rewards += weights[11] * (speed_ratio - passive_pen)

        # ── R13: BallAcceleration (reward hitting ball hard on touch) ──
        if weights[12] > 0:
            ball_accel = np.maximum(0.0, ball_speed_1d - self.prev_ball_speed)  # (E,)
            accel_norm = ball_accel / BALL_MAX_SPEED  # (E,) normalized [0, 1]
            rewards += weights[12] * (accel_norm[:, None] * bs.player_ball_touched)

        # ── Event Reward ──
        self._current_ev[:] = 0.0
        current_ev = self._current_ev
        current_ev[:, :, 0] = bs.player_goals
        for j in range(A):
            blue_mask = (bs.player_team[:, j] == BLUE_TEAM)
            current_ev[:, j, 1] = np.where(blue_mask, bs.blue_score, bs.orange_score)
            current_ev[:, j, 2] = np.where(blue_mask, bs.orange_score, bs.blue_score)
        current_ev[:, :, 3] = bs.player_ball_touched
        current_ev[:, :, 4] = bs.player_shots
        current_ev[:, :, 5] = bs.player_saves
        current_ev[:, :, 6] = bs.player_demos
        current_ev[:, :, 7] = bs.player_boost
        diff = np.maximum(current_ev - self.event_last, 0.0)
        rewards += np.sum(diff * ev_weights[None, None, :], axis=2)
        self.event_last[:] = current_ev

        # ── Team Spirit blending ──
        if ts > 0:
            blue_mean = (rewards[:, 0] + rewards[:, 1]) * 0.5
            orange_mean = (rewards[:, 2] + rewards[:, 3]) * 0.5
            team_mean = np.empty_like(rewards)
            team_mean[:, 0] = blue_mean
            team_mean[:, 1] = blue_mean
            team_mean[:, 2] = orange_mean
            team_mean[:, 3] = orange_mean
            rewards = (1.0 - ts) * rewards + ts * team_mean

        # ── Update tracking state ──
        self.prev_player_boost[:] = bs.player_boost
        self.prev_ball_speed[:] = ball_speed_1d

        return rewards


# ─────────────────────────────────────────────────────────
# VECTORIZED OBS BUILDING
# ─────────────────────────────────────────────────────────

def build_obs_batch(bs, prev_actions):
    """Build observations for all envs/agents in vectorized numpy.

    Matches DefaultObs format: ball(9) + prev_action(8) + pads(34) +
    self(19) + ally(19) + enemy0(19) + enemy1(19) = 127

    Args:
        bs: BatchState
        prev_actions: (E, A, 8) previous actions per agent

    Returns: (E*A, 127) observation array
    """
    E, A = bs.n_envs, bs.n_agents
    obs = np.empty((E, A, 127), dtype=np.float32)

    # ── Ball section (0-8) ──
    # Blue agents (0,1): normal ball; Orange agents (2,3): inverted
    obs[:, :2, 0:3] = bs.ball_pos[:, None, :] * POS_COEF
    obs[:, :2, 3:6] = bs.ball_vel[:, None, :] * LIN_VEL_COEF
    obs[:, :2, 6:9] = bs.ball_ang[:, None, :] * ANG_VEL_COEF
    obs[:, 2:, 0:3] = bs.inv_ball_pos[:, None, :] * POS_COEF
    obs[:, 2:, 3:6] = bs.inv_ball_vel[:, None, :] * LIN_VEL_COEF
    obs[:, 2:, 6:9] = bs.inv_ball_ang[:, None, :] * ANG_VEL_COEF

    # ── Previous actions (9-16) ──
    obs[:, :, 9:17] = prev_actions

    # ── Boost pads (17-50) ──
    obs[:, :2, 17:51] = bs.boost_pads[:, None, :]
    obs[:, 2:, 17:51] = bs.inv_boost_pads[:, None, :]

    # ── Helper: fill 19-element player block ──
    def fill_block(sl, pos, fwd, up, vel, ang, boost, ground, flip, demoed):
        sl[:, :, 0:3] = pos * POS_COEF
        sl[:, :, 3:6] = fwd
        sl[:, :, 6:9] = up
        sl[:, :, 9:12] = vel * LIN_VEL_COEF
        sl[:, :, 12:15] = ang * ANG_VEL_COEF
        sl[:, :, 15] = boost
        sl[:, :, 16] = ground
        sl[:, :, 17] = flip
        sl[:, :, 18] = demoed

    # Shorthand arrays for blue/orange views
    pp, pf, pu, pv, pa = bs.player_pos, bs.player_fwd, bs.player_up, bs.player_vel, bs.player_ang_vel
    ip, if_, iu, iv, ia = bs.inv_player_pos, bs.inv_player_fwd, bs.inv_player_up, bs.inv_player_vel, bs.inv_player_ang_vel
    pb, pg, pfl, pd = bs.player_boost, bs.player_on_ground, bs.player_has_flip, bs.player_is_demoed

    # ── Self (51-69) ──
    fill_block(obs[:, :2, 51:70], pp[:, :2], pf[:, :2], pu[:, :2], pv[:, :2], pa[:, :2],
               pb[:, :2], pg[:, :2], pfl[:, :2], pd[:, :2])
    fill_block(obs[:, 2:, 51:70], ip[:, 2:], if_[:, 2:], iu[:, 2:], iv[:, 2:], ia[:, 2:],
               pb[:, 2:], pg[:, 2:], pfl[:, 2:], pd[:, 2:])

    # ── Ally (70-88): ally indices [1,0,3,2] ──
    fill_block(obs[:, :2, 70:89], pp[:, ALLY_IDX[:2]], pf[:, ALLY_IDX[:2]], pu[:, ALLY_IDX[:2]],
               pv[:, ALLY_IDX[:2]], pa[:, ALLY_IDX[:2]],
               pb[:, ALLY_IDX[:2]], pg[:, ALLY_IDX[:2]], pfl[:, ALLY_IDX[:2]], pd[:, ALLY_IDX[:2]])
    fill_block(obs[:, 2:, 70:89], ip[:, ALLY_IDX[2:]], if_[:, ALLY_IDX[2:]], iu[:, ALLY_IDX[2:]],
               iv[:, ALLY_IDX[2:]], ia[:, ALLY_IDX[2:]],
               pb[:, ALLY_IDX[2:]], pg[:, ALLY_IDX[2:]], pfl[:, ALLY_IDX[2:]], pd[:, ALLY_IDX[2:]])

    # ── Enemy 0 (89-107): enemy0 indices [2,2,0,0] ──
    fill_block(obs[:, :2, 89:108], pp[:, ENEMY0_IDX[:2]], pf[:, ENEMY0_IDX[:2]], pu[:, ENEMY0_IDX[:2]],
               pv[:, ENEMY0_IDX[:2]], pa[:, ENEMY0_IDX[:2]],
               pb[:, ENEMY0_IDX[:2]], pg[:, ENEMY0_IDX[:2]], pfl[:, ENEMY0_IDX[:2]], pd[:, ENEMY0_IDX[:2]])
    fill_block(obs[:, 2:, 89:108], ip[:, ENEMY0_IDX[2:]], if_[:, ENEMY0_IDX[2:]], iu[:, ENEMY0_IDX[2:]],
               iv[:, ENEMY0_IDX[2:]], ia[:, ENEMY0_IDX[2:]],
               pb[:, ENEMY0_IDX[2:]], pg[:, ENEMY0_IDX[2:]], pfl[:, ENEMY0_IDX[2:]], pd[:, ENEMY0_IDX[2:]])

    # ── Enemy 1 (108-126): enemy1 indices [3,3,1,1] ──
    fill_block(obs[:, :2, 108:127], pp[:, ENEMY1_IDX[:2]], pf[:, ENEMY1_IDX[:2]], pu[:, ENEMY1_IDX[:2]],
               pv[:, ENEMY1_IDX[:2]], pa[:, ENEMY1_IDX[:2]],
               pb[:, ENEMY1_IDX[:2]], pg[:, ENEMY1_IDX[:2]], pfl[:, ENEMY1_IDX[:2]], pd[:, ENEMY1_IDX[:2]])
    fill_block(obs[:, 2:, 108:127], ip[:, ENEMY1_IDX[2:]], if_[:, ENEMY1_IDX[2:]], iu[:, ENEMY1_IDX[2:]],
               iv[:, ENEMY1_IDX[2:]], ia[:, ENEMY1_IDX[2:]],
               pb[:, ENEMY1_IDX[2:]], pg[:, ENEMY1_IDX[2:]], pfl[:, ENEMY1_IDX[2:]], pd[:, ENEMY1_IDX[2:]])

    return obs.reshape(E * A, 127)


# ─────────────────────────────────────────────────────────
# VECTORIZED TERMINAL CONDITIONS
# ─────────────────────────────────────────────────────────

class VectorizedTerminals:
    """Tracks GoalScoredCondition + TimeoutCondition per env."""

    def __init__(self, n_envs, max_steps=300):
        self.n_envs = n_envs
        self.max_steps = max_steps
        self.step_counts = np.zeros(n_envs, dtype=np.int32)
        self.last_blue = np.zeros(n_envs, dtype=np.int32)
        self.last_orange = np.zeros(n_envs, dtype=np.int32)

    def reset_env(self, env_idx, bs):
        self.step_counts[env_idx] = 0
        self.last_blue[env_idx] = bs.blue_score[env_idx]
        self.last_orange[env_idx] = bs.orange_score[env_idx]

    def check(self, bs):
        """Check terminal conditions for all envs. Returns (E,) bool array."""
        self.step_counts += 1
        timeout = self.step_counts >= self.max_steps
        goal = ((bs.blue_score != self.last_blue) |
                (bs.orange_score != self.last_orange))
        # Update tracked scores
        self.last_blue[:] = bs.blue_score
        self.last_orange[:] = bs.orange_score
        return timeout | goal
