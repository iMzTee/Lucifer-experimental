"""collector.py — GPUCollector: drop-in replacement for VectorizedCollector.

Same interface as vectorized_collector.py but everything stays on GPU.
No CPU↔GPU transfers in the hot loop — only at the end when returning
experience to the PPO learner.
"""

import torch
import time
import json
import os
from .environment import GPUEnvironment
from .rewards import GPURewards
from .observations import build_obs_batch
from .constants import STAGE_CONFIG


class WelfordRunningStatGPU:
    """Running mean/std stats on GPU for observation normalization."""

    def __init__(self, shape, device='cuda'):
        self.shape = shape
        self.device = device
        self.count = 0
        self.mean = torch.zeros(shape, device=device)
        self.M2 = torch.zeros(shape, device=device)

    @property
    def std(self):
        if self.count < 2:
            return torch.ones(self.shape, device=self.device)
        variance = self.M2 / (self.count - 1)
        return torch.sqrt(variance).clamp(min=1e-6)

    def increment(self, batch, count):
        """Update stats with a batch of observations. batch: (N, obs_size)."""
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False) if count > 1 else torch.zeros_like(batch_mean)

        new_count = self.count + count
        delta = batch_mean - self.mean
        self.mean += delta * (count / new_count)
        self.M2 += batch_var * count + delta ** 2 * self.count * count / new_count
        self.count = new_count


class GPUCollector:
    """Drop-in replacement for VectorizedCollector. Same interface.

    Everything stays on GPU:
    - Physics simulation (GPUEnvironment)
    - Reward computation (GPURewards)
    - Observation building (build_obs_batch)
    - Policy inference
    - Experience buffer

    Only moves to CPU at the very end when returning final experience tuple.
    """

    def __init__(self, n_envs, device='cuda', standardize_obs=True, stage=0):
        self.n_envs = n_envs
        self.device = device
        self.n_agents = 4  # 2v2
        self.total_agents = n_envs * 4
        self.obs_size = 127
        self.act_size = 8  # MultiDiscrete(3,3,3,3,3,2,2,2)

        self.policy = None
        self.cumulative_timesteps = 0
        self.standardize_obs = standardize_obs

        self._stage = stage
        self.env = GPUEnvironment(n_envs, device, stage=stage)
        self.rewards = GPURewards(n_envs, device)

        # Obs normalization
        if standardize_obs:
            self.obs_stats = WelfordRunningStatGPU((self.obs_size,), device)
        else:
            self.obs_stats = None
        self._stats_step_counter = 0
        self._steps_per_stats_increment = 5

        # Previous actions (on GPU)
        self._prev_actions = torch.zeros(n_envs, 4, 8, device=device)

        # Current observation (on GPU)
        self._current_obs = torch.zeros(self.total_agents, self.obs_size, device=device)

        # Initialize
        self.env.reset_all()
        self._current_obs = build_obs_batch(self.env.state, self._prev_actions)
        self.rewards.reset_envs(
            torch.ones(n_envs, dtype=torch.bool, device=device), self.env.state)

        if standardize_obs:
            self.obs_stats.increment(self._current_obs, self.total_agents)

        print(f"[GPUCollector] Ready: {n_envs} envs x 4 agents = {self.total_agents} "
              f"agents/step, obs_size={self.obs_size}, stage={stage}, device={device}")

    @property
    def shapes(self):
        """Returns (obs_size, act_size, policy_type).
        policy_type=1 for MultiDiscrete."""
        return (self.obs_size, self.act_size, 1)

    def set_stage(self, stage):
        """Update curriculum stage."""
        self._stage = stage
        self.env.set_stage(stage)

    def _normalize_obs(self, obs):
        """Normalize obs using running stats. obs: (N, 127) GPU tensor."""
        if not self.standardize_obs:
            return obs
        return ((obs - self.obs_stats.mean) / self.obs_stats.std).clamp(-5.0, 5.0)

    def collect_timesteps(self, n):
        """Collect at least n timesteps across all envs.

        Returns: (exp_tuple, metrics_list, steps_collected, elapsed_time)
        where exp_tuple = (states, actions, log_probs, rewards, next_states, dones, truncated)
        All returned as numpy arrays on CPU (compatible with existing PPO learner).
        """
        assert self.policy is not None, "Set .policy before calling collect_timesteps()"
        t0 = time.time()

        n_total = self.total_agents
        n_rounds = (n + n_total - 1) // n_total

        # Pre-allocate GPU buffers for experience
        max_steps = n_rounds * n_total
        all_states = torch.empty(max_steps, self.obs_size, device=self.device)
        all_actions = torch.empty(max_steps, self.act_size, device=self.device)
        all_log_probs = torch.empty(max_steps, device=self.device)
        all_rewards = torch.empty(max_steps, device=self.device)
        all_next_states = torch.empty(max_steps, self.obs_size, device=self.device)
        all_dones = torch.zeros(max_steps, device=self.device)
        all_truncated = torch.zeros(max_steps, device=self.device)

        write_idx = 0
        steps_collected = 0

        while steps_collected < n:
            # ── 1. Normalize current obs ──
            obs_norm = self._normalize_obs(self._current_obs)

            # ── 2. Policy inference (on GPU) ──
            with torch.no_grad():
                actions_t, log_probs_t = self.policy.get_action(obs_norm)

            # Save pre-step states
            pre_states = obs_norm

            # ── 3. Parse discrete actions to continuous controls ──
            # actions_t: (E*4, 8) integer tensor from MultiDiscrete
            # Convert: first 5 dims subtract 1 to get [-1, 0, 1], last 3 are {0, 1}
            actions_float = actions_t.float()
            controls = torch.zeros(self.n_envs, 4, 8, device=self.device)
            actions_reshaped = actions_float.reshape(self.n_envs, 4, 8)
            controls[:, :, :5] = actions_reshaped[:, :, :5] - 1.0
            controls[:, :, 5:] = actions_reshaped[:, :, 5:]
            self._prev_actions = controls.clone()

            # ── 4. Step environment ──
            terminals = self.env.step(controls)

            # ── 5. Compute rewards ──
            rewards_batch = self.rewards.compute(self.env.state, self._stage)  # (E, 4)

            # ── 6. Build next observations ──
            next_obs = build_obs_batch(self.env.state, self._prev_actions)

            # ── 7. Update obs stats ──
            self._stats_step_counter += 1
            if (self.standardize_obs and
                    self._stats_step_counter >= self._steps_per_stats_increment):
                self.obs_stats.increment(next_obs, n_total)
                self._stats_step_counter = 0

            # ── 8. Normalize next obs ──
            next_norm = self._normalize_obs(next_obs)

            # ── 9. Record experience ──
            end_idx = write_idx + n_total
            all_states[write_idx:end_idx] = pre_states
            all_actions[write_idx:end_idx] = actions_float
            all_log_probs[write_idx:end_idx] = log_probs_t
            all_rewards[write_idx:end_idx] = rewards_batch.reshape(-1)
            all_next_states[write_idx:end_idx] = next_norm
            dones_expanded = terminals.unsqueeze(1).expand(-1, 4).reshape(-1).float()
            all_dones[write_idx:end_idx] = dones_expanded

            write_idx = end_idx
            steps_collected += n_total

            # ── 10. Update current obs + reset done envs ──
            self._current_obs = next_obs

            if terminals.any():
                # Reset done envs
                self.env.reset_done_envs(terminals)
                # Reset rewards state for done envs
                self.rewards.reset_envs(terminals, self.env.state)
                # Rebuild obs for reset envs
                new_obs = build_obs_batch(self.env.state, self._prev_actions)
                # Only update obs for reset envs
                reset_agents = terminals.unsqueeze(1).expand(-1, 4).reshape(-1)
                self._current_obs[reset_agents] = new_obs[reset_agents]
                # Zero prev_actions for reset envs
                self._prev_actions[terminals] = 0.0

                if self.standardize_obs:
                    self.obs_stats.increment(new_obs[reset_agents], reset_agents.sum().item())

        # Trim to actual size
        actual = write_idx
        all_states = all_states[:actual]
        all_actions = all_actions[:actual]
        all_log_probs = all_log_probs[:actual]
        all_rewards = all_rewards[:actual]
        all_next_states = all_next_states[:actual]
        all_dones = all_dones[:actual]
        all_truncated = all_truncated[:actual]

        # Mark last round non-done transitions as truncated (for GAE)
        last_round_start = actual - n_total
        if last_round_start >= 0:
            for i in range(self.n_envs):
                base = last_round_start + i * self.n_agents
                if base < actual and all_dones[base] == 0:
                    for j in range(self.n_agents):
                        if base + j < actual:
                            all_truncated[base + j] = 1.0

        self.cumulative_timesteps += actual
        elapsed = time.time() - t0

        # ── Move to CPU as numpy (compatible with existing PPO learner) ──
        exp = (
            all_states.cpu().numpy().astype('float32'),
            all_actions.cpu().numpy().astype('float32'),
            all_log_probs.cpu().numpy().astype('float32'),
            all_rewards.cpu().numpy().astype('float32'),
            all_next_states.cpu().numpy().astype('float32'),
            all_dones.cpu().numpy().astype('float32'),
            all_truncated.cpu().numpy().astype('float32'),
        )

        return exp, [], actual, elapsed

    # Cleanup methods (no-ops — no child processes)
    def stop(self): pass
    def terminate(self): pass
    def shutdown(self): pass
    def close(self): pass
    def cleanup(self): pass
