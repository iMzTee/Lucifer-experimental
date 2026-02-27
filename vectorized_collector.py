"""vectorized_collector.py — Single-process vectorized timestep collector.

Replaces rlgym_ppo.batched_agents.BatchedAgentManager with a single-process
approach using Arena.multi_step() for parallel C++ physics simulation.
Bypasses _build_gamestate() — extracts directly from raw arena state data.

Drop-in interface: .policy, .cumulative_timesteps, .collect_timesteps(n)
"""

import numpy as np
import torch
import time
import json
import os
import RocketSim as rsim
from rlgym_ppo.util.running_stats import WelfordRunningStat
from vectorized_env import (
    BatchState, VectorizedRewards, build_obs_batch,
    VectorizedTerminals, STAGE_CONFIG, JUMP_TIMER_SECONDS,
)

_HAS_BATCH_API = hasattr(rsim.Arena, 'multi_set_controls')

try:
    import fast_step
    _HAS_FAST_STEP = True
except ImportError:
    _HAS_FAST_STEP = False


class VectorizedCollector:
    """Single-process vectorized step collector using Arena.multi_step().

    Bypasses rlgym_sim's _build_gamestate() for the hot path — extracts state
    directly from arena.get_gym_state() raw numpy arrays for maximum speed.
    """

    def __init__(self, build_env_fn, n_envs, standardize_obs=True,
                 steps_per_obs_stats_increment=5):
        self.n_envs = n_envs
        self.policy = None
        self.cumulative_timesteps = 0
        self.standardize_obs = standardize_obs
        self._stats_step_counter = 0
        self._steps_per_stats_increment = steps_per_obs_stats_increment

        # Create environments
        print(f"[VecCollector] Creating {n_envs} environments...")
        t0 = time.time()
        self.envs = []
        for i in range(n_envs):
            self.envs.append(build_env_fn())
            if (i + 1) % 10 == 0 or i == n_envs - 1:
                print(f"[VecCollector]   {i + 1}/{n_envs} created")
        print(f"[VecCollector] Envs created in {time.time() - t0:.1f}s")

        # Extract env properties
        env0 = self.envs[0]
        self.obs_size = int(np.prod(env0.observation_space.shape))
        self.n_agents_per_env = env0._match.agents
        self.total_agents = self.n_envs * self.n_agents_per_env
        n_ag = self.n_agents_per_env

        # Derive action size from action space
        if hasattr(env0.action_space, 'nvec'):
            self.act_size = len(env0.action_space.nvec)  # MultiDiscrete
        else:
            self.act_size = int(np.prod(env0.action_space.shape))

        # Cache internal references for fast access
        self._games = [e._game for e in self.envs]
        self._matches = [e._match for e in self.envs]
        self._arenas = [g.arena for g in self._games]
        self.tick_skip = self._games[0].tick_skip

        # Build car ordering maps + car object references for direct extraction
        self._car_orders = []   # per-game: np.array[arena_k] → ordered_idx
        self._car_objects = []  # per-game: list of car objects in ordered order
        self._game_scores = []  # per-game: [blue_score, orange_score] for goal detection
        for game in self._games:
            raw = game.arena.get_gym_state()
            n_players = len(raw) - 3
            order_map = np.empty(n_players, dtype=np.int32)
            for k in range(n_players):
                raw_car_id = int(raw[3 + k][0][0])
                spec_id = game.car_id_to_spectator_map[raw_car_id]
                order_map[k] = game.spectator_to_ordered_list_map[spec_id]
            self._car_orders.append(order_map)

            # Get car objects in ordered order
            cars_ordered = [None] * n_players
            for car_id, player in game.players.items():
                spec_id = game.car_id_to_spectator_map[car_id]
                idx = game.spectator_to_ordered_list_map[spec_id]
                cars_ordered[idx] = player.car
            self._car_objects.append(cars_ordered)
            self._game_scores.append([game.blue_score, game.orange_score])

        # Batch API / fast step acceleration structures
        if _HAS_BATCH_API or _HAS_FAST_STEP:
            self._car_objects_flat = [car for env_cars in self._car_objects
                                     for car in env_cars]
            self._car_orders_np = np.ascontiguousarray(
                np.array(self._car_orders, dtype=np.int32))
            self._game_scores_np = np.array(
                self._game_scores, dtype=np.int32)

        if _HAS_BATCH_API:
            self._controls_buf = np.zeros(
                (n_envs, n_ag, 8), dtype=np.float32)

        # Vectorized computation modules
        self._batch_state = BatchState(n_envs, n_ag)
        self._vec_rewards = VectorizedRewards(n_envs, n_ag)
        # Load curriculum stage (must happen before VectorizedTerminals init)
        self._load_stage()

        cfg = STAGE_CONFIG.get(self._stage, STAGE_CONFIG[0])
        self._vec_terminals = VectorizedTerminals(n_envs, max_steps=cfg["timeout"])

        # Obs normalization
        if standardize_obs:
            self.obs_stats = WelfordRunningStat((self.obs_size,))
        else:
            self.obs_stats = None

        # Previous actions tracker: (E, A, 8) — starts as zeros
        self._prev_actions = np.zeros((n_envs, n_ag, 8), dtype=np.float32)

        # Current obs stored as flat (total_agents, obs_size) array
        self._current_obs = np.empty((self.total_agents, self.obs_size), dtype=np.float32)

        # Reset all envs and store initial observations
        for i, env in enumerate(self.envs):
            obs = env.reset()
            obs = obs if isinstance(obs, list) else [obs]
            for j, o in enumerate(obs):
                self._current_obs[i * n_ag + j] = o
            if standardize_obs:
                self.obs_stats.increment(np.array(obs), len(obs))

        # Initialize batch state from post-reset env states
        game_states = [env._prev_state for env in self.envs]
        self._batch_state.extract(game_states)
        for i in range(n_envs):
            self._vec_terminals.reset_env(i, self._batch_state)
            self._vec_rewards.reset_env(i, self._batch_state)

        print(f"[VecCollector] Ready: {n_envs} envs x {n_ag} agents "
              f"= {self.total_agents} agents/step, tick_skip={self.tick_skip}, "
              f"obs_size={self.obs_size}, stage={self._stage}")

    def _load_stage(self):
        """Load current curriculum stage from curriculum_state.json."""
        state_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "curriculum_state.json")
        if os.path.exists(state_file):
            with open(state_file) as f:
                state = json.load(f)
            self._stage = state.get("stage", 0)
        else:
            self._stage = 0

    def set_stage(self, stage):
        """Update curriculum stage (called from Learner when stage changes)."""
        self._stage = stage

    @property
    def shapes(self):
        """Returns (obs_size, act_size, policy_type) matching init_processes."""
        import gym.spaces
        env = self.envs[0]
        obs_size = int(np.prod(env.observation_space.shape))
        act_size = int(np.prod(env.action_space.shape))
        t = type(env.action_space)
        if t == gym.spaces.multi_discrete.MultiDiscrete:
            action_space_type = 1
        elif t == gym.spaces.box.Box:
            action_space_type = 2
        else:
            action_space_type = 0
        return (obs_size, act_size, action_space_type)

    def _normalize_obs_batch(self, obs_batch):
        """Normalize batch of obs (N, obs_size) using running stats. Clip to [-5, 5]."""
        if not self.standardize_obs:
            return obs_batch
        mean = self.obs_stats.mean.astype(np.float32)
        std = self.obs_stats.std.astype(np.float32)
        return np.clip((obs_batch - mean) / std, -5.0, 5.0)

    def collect_timesteps(self, n):
        """Collect at least n timesteps across all envs.

        Returns: (exp_tuple, metrics_list, steps_collected, elapsed_time)
        """
        assert self.policy is not None, "Set .policy before calling collect_timesteps()"
        t0 = time.time()
        n_ag = self.n_agents_per_env
        n_total = self.total_agents  # E * A

        # Pre-allocate arrays
        n_rounds = (n + n_total - 1) // n_total
        max_steps = n_rounds * n_total

        all_states = np.empty((max_steps, self.obs_size), dtype=np.float16)
        all_actions = np.empty((max_steps, self.act_size), dtype=np.int8)
        all_log_probs = np.empty(max_steps, dtype=np.float32)
        all_rewards = np.empty(max_steps, dtype=np.float32)
        all_next_states = np.empty((max_steps, self.obs_size), dtype=np.float16)
        all_dones = np.zeros(max_steps, dtype=np.bool_)
        all_truncated = np.zeros(max_steps, dtype=np.bool_)

        write_idx = 0
        steps_collected = 0

        while steps_collected < n:
            # ─── Phase 1: Normalize obs + GPU policy inference ───
            obs_norm = self._normalize_obs_batch(self._current_obs)

            with torch.no_grad():
                actions_t, log_probs_t = self.policy.get_action(obs_norm)
            actions_np = actions_t.cpu().numpy() if actions_t.is_cuda else actions_t.numpy()
            log_probs_np = log_probs_t.cpu().numpy() if log_probs_t.is_cuda else log_probs_t.numpy()

            # Save pre-step states (normalized)
            pre_states = obs_norm

            if _HAS_BATCH_API:
                # ─── Native RocketSim batch API (2 boundary crossings) ───
                # Parse actions into controls buffer
                acts = actions_np.reshape(
                    self.n_envs, n_ag, -1).astype(np.float32)
                self._controls_buf[:, :, :5] = acts[:, :, :5] - 1
                self._controls_buf[:, :, 5:] = acts[:, :, 5:]

                # Set controls (1 C++ call)
                rsim.Arena.multi_set_controls(
                    self._arenas, self._controls_buf)

                # Track prev_actions
                self._prev_actions[:] = self._controls_buf

                # Physics + goal detection + extraction + has_flip (1 C++ call)
                goal_envs = rsim.Arena.multi_get_gym_state(
                    self._arenas, self._car_orders_np,
                    self._batch_state, self._game_scores_np,
                    self.tick_skip, JUMP_TIMER_SECONDS)

                # Sync game scores for goal envs
                goal_indices = np.where(goal_envs)[0]
                for i in goal_indices:
                    self._games[i].blue_score = int(
                        self._game_scores_np[i, 0])
                    self._games[i].orange_score = int(
                        self._game_scores_np[i, 1])

            elif _HAS_FAST_STEP:
                # ─── fast_step pybind11 path (legacy fallback) ───
                actions_i32 = np.ascontiguousarray(actions_np, dtype=np.int32)
                fast_step.fast_set_controls(
                    self._car_objects_flat, actions_i32,
                    self.n_envs, n_ag)

                self._prev_actions[:] = actions_np.reshape(
                    self.n_envs, n_ag, -1).astype(np.float32)
                self._prev_actions[:, :, :5] -= 1

                goal_envs = fast_step.fast_get_and_extract(
                    self._arenas, self._car_orders_np,
                    self._car_objects_flat, self._batch_state,
                    self._game_scores_np, self.tick_skip,
                    JUMP_TIMER_SECONDS)

                goal_indices = np.where(goal_envs)[0]
                for i in goal_indices:
                    self._games[i].blue_score = int(
                        self._game_scores_np[i, 0])
                    self._games[i].orange_score = int(
                        self._game_scores_np[i, 1])
            else:
                # ─── Python fallback (Phases 2-7) ───
                # Phase 2-3: Parse/format actions, set controls per env
                agent_idx = 0
                for i in range(self.n_envs):
                    env_actions = actions_np[agent_idx:agent_idx + n_ag]
                    parsed = self._matches[i].parse_actions(
                        env_actions, self.envs[i]._prev_state)
                    formatted = self._matches[i].format_actions(parsed)
                    self._games[i]._set_controls(formatted)
                    # Capture prev_actions (format_actions stores them)
                    self._prev_actions[i] = \
                        self._matches[i]._prev_actions[:n_ag]
                    agent_idx += n_ag

                # Phase 4: Parallel first physics tick
                rsim.Arena.multi_step(self._arenas, 1)

                # Phase 5: Get raw state + handle goal resets
                raw_states = [arena.get_gym_state()
                              for arena in self._arenas]
                for i, raw in enumerate(raw_states):
                    bs_score = int(raw[0][2])
                    os_score = int(raw[0][3])
                    if (bs_score != self._game_scores[i][0] or
                            os_score != self._game_scores[i][1]):
                        self._game_scores[i][0] = bs_score
                        self._game_scores[i][1] = os_score
                        self._games[i].blue_score = bs_score
                        self._games[i].orange_score = os_score
                        self._arenas[i].ball.set_state(rsim.BallState())

                # Phase 6: Parallel remaining physics ticks
                if self.tick_skip > 1:
                    rsim.Arena.multi_step(self._arenas,
                                          self.tick_skip - 1)

                # Phase 7: Vectorized extract
                self._batch_state.extract_raw(
                    raw_states, self._car_orders, self._car_objects)

            # Vectorized terminal check → (E,) bool
            dones_batch = self._vec_terminals.check(self._batch_state)

            # Vectorized rewards → (E, A)
            rewards_batch = self._vec_rewards.compute(
                self._batch_state, self._stage)

            # Vectorized obs building → (E*A, 127)
            next_obs = build_obs_batch(self._batch_state, self._prev_actions)

            # Update obs stats periodically
            self._stats_step_counter += 1
            if (self.standardize_obs and
                    self._stats_step_counter >= self._steps_per_stats_increment):
                self.obs_stats.increment(next_obs, n_total)
                self._stats_step_counter = 0

            # Normalize next observations
            next_norm = self._normalize_obs_batch(next_obs)

            # ─── Record transitions (vectorized, compact dtypes) ───
            end_idx = write_idx + n_total
            all_states[write_idx:end_idx] = pre_states.astype(np.float16)
            all_actions[write_idx:end_idx] = actions_np.astype(np.int8)
            all_log_probs[write_idx:end_idx] = log_probs_np
            all_rewards[write_idx:end_idx] = rewards_batch.ravel()
            all_next_states[write_idx:end_idx] = next_norm.astype(np.float16)
            all_dones[write_idx:end_idx] = np.repeat(dones_batch, n_ag)

            write_idx = end_idx
            steps_collected += n_total

            # ─── Phase 8: Update current obs + reset done envs ───
            self._current_obs[:] = next_obs

            done_indices = np.where(dones_batch)[0]
            for i in done_indices:
                obs_reset = self.envs[i].reset()
                obs_reset = obs_reset if isinstance(obs_reset, list) else [obs_reset]
                base = i * n_ag
                for j, o in enumerate(obs_reset):
                    self._current_obs[base + j] = o
                self._prev_actions[i] = 0
                # Update game scores tracker after reset
                self._game_scores[i][0] = self._games[i].blue_score
                self._game_scores[i][1] = self._games[i].orange_score
                if _HAS_BATCH_API or _HAS_FAST_STEP:
                    self._game_scores_np[i, 0] = self._games[i].blue_score
                    self._game_scores_np[i, 1] = self._games[i].orange_score
                if self.standardize_obs:
                    self.obs_stats.increment(np.array(obs_reset), len(obs_reset))
                # Update batch state for this env after reset
                self._batch_state.extract_single(
                    i, self.envs[i]._prev_state)
                self._vec_terminals.reset_env(i, self._batch_state)
                self._vec_rewards.reset_env(i, self._batch_state)

        # Trim to actual size
        actual = write_idx
        all_states = all_states[:actual]
        all_actions = all_actions[:actual]
        all_log_probs = all_log_probs[:actual]
        all_rewards = all_rewards[:actual]
        all_next_states = all_next_states[:actual]
        all_dones = all_dones[:actual]
        all_truncated = all_truncated[:actual]

        # Mark last round's non-done transitions as truncated (for GAE)
        last_round_start = actual - n_total
        if last_round_start >= 0:
            for i in range(self.n_envs):
                base = last_round_start + i * n_ag
                if base < actual and not all_dones[base]:
                    for j in range(n_ag):
                        if base + j < actual:
                            all_truncated[base + j] = True

        self.cumulative_timesteps += actual
        elapsed = time.time() - t0

        # Cast compact arrays back to float32 in chunks to avoid doubling memory
        CAST_CHUNK = 50000
        out_states = np.empty_like(all_states, dtype=np.float32)
        out_next = np.empty_like(all_next_states, dtype=np.float32)
        out_acts = np.empty((actual, self.act_size), dtype=np.float32)
        out_dones = np.empty(actual, dtype=np.float32)
        out_trunc = np.empty(actual, dtype=np.float32)
        for ci in range(0, actual, CAST_CHUNK):
            ce = min(ci + CAST_CHUNK, actual)
            out_states[ci:ce] = all_states[ci:ce].astype(np.float32)
            out_next[ci:ce] = all_next_states[ci:ce].astype(np.float32)
            out_acts[ci:ce] = all_actions[ci:ce].astype(np.float32)
            out_dones[ci:ce] = all_dones[ci:ce].astype(np.float32)
            out_trunc[ci:ce] = all_truncated[ci:ce].astype(np.float32)
        del all_states, all_next_states, all_actions, all_dones, all_truncated

        exp = (out_states, out_acts, all_log_probs, all_rewards,
               out_next, out_dones, out_trunc)
        return exp, [], actual, elapsed

    # Cleanup methods (no-ops — no child processes to kill)
    def stop(self): pass
    def terminate(self): pass
    def shutdown(self): pass
    def close(self): pass
    def cleanup(self): pass
