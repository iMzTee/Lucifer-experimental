"""collector.py — GPUCollector: drop-in replacement for VectorizedCollector.

Same interface as vectorized_collector.py but everything stays on GPU.
No CPU↔GPU transfers in the hot loop — only at the end when returning
experience to the PPO learner.

Supports variable n_agents (1v0, 1v1, 2v2) via STAGE_CONFIG.
"""

import torch
import time
from .environment import GPUEnvironment
from .rewards import GPURewards
from .observations import build_obs_batch
from .constants import STAGE_CONFIG, get_agent_layout


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
    """Drop-in replacement for VectorizedCollector.

    Supports variable n_agents per stage. Full re-init on stage change
    when n_agents or n_envs changes.
    """

    def __init__(self, n_envs, device='cuda', standardize_obs=True, stage=0, vis_sender=None):
        self.device = device
        self.obs_size = 127
        self.act_size = 8  # MultiDiscrete(5,5,5,5,5,2,2,2)
        self.policy = None
        self.cumulative_timesteps = 0
        self.standardize_obs = standardize_obs
        self._stage = stage
        self.vis_sender = vis_sender

        # Get config for stage
        cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[0])
        self.n_envs = n_envs
        self.n_agents = cfg.get("n_agents", 4)
        self.layout = get_agent_layout(self.n_agents)
        self.total_agents = n_envs * self.n_agents

        self.env = GPUEnvironment(n_envs, device, stage=stage)
        if vis_sender is not None and vis_sender.enabled:
            self.env.tick_skip = 2  # smoother vis while keeping physics sane
        self.rewards = GPURewards(n_envs, device, n_agents=self.n_agents, layout=self.layout)

        # Obs normalization
        if standardize_obs:
            self.obs_stats = WelfordRunningStatGPU((self.obs_size,), device)
        else:
            self.obs_stats = None
        self._stats_step_counter = 0
        self._steps_per_stats_increment = 5

        # Previous actions and controls (on GPU)
        A = self.n_agents
        self._prev_actions = torch.zeros(n_envs, A, 8, device=device)
        self._controls = torch.zeros(n_envs, A, 8, device=device)

        # Current observation (on GPU)
        self._current_obs = torch.zeros(self.total_agents, self.obs_size, device=device)

        # Initialize
        self.env.reset_all()
        self._current_obs = build_obs_batch(self.env.state, self._prev_actions)
        self.rewards.reset_envs(
            torch.ones(n_envs, dtype=torch.bool, device=device), self.env.state)

        if standardize_obs:
            self.obs_stats.increment(self._current_obs, self.total_agents)

        print(f"[GPUCollector] Ready: {n_envs} envs x {A} agents = {self.total_agents} "
              f"agents/step, obs_size={self.obs_size}, stage={stage}, device={device}")

    @property
    def shapes(self):
        """Returns (obs_size, act_size, policy_type).
        policy_type=1 for MultiDiscrete."""
        return (self.obs_size, self.act_size, 1)

    def set_stage(self, stage):
        """Update curriculum stage. Full re-init if n_agents or n_envs changes."""
        cfg = STAGE_CONFIG.get(stage, STAGE_CONFIG[0])
        new_n_agents = cfg.get("n_agents", 4)
        new_n_envs = cfg.get("n_envs", self.n_envs)
        if self.vis_sender is not None and self.vis_sender.enabled:
            new_n_envs = min(new_n_envs, 3000)

        if new_n_agents != self.n_agents or new_n_envs != self.n_envs:
            # Full re-init needed
            old_policy = self.policy
            old_obs_stats = self.obs_stats
            old_cumul = self.cumulative_timesteps

            self._stage = stage
            self.n_envs = new_n_envs
            self.n_agents = new_n_agents
            self.layout = get_agent_layout(new_n_agents)
            self.total_agents = new_n_envs * new_n_agents

            self.env = GPUEnvironment(new_n_envs, self.device, stage=stage)
            if self.vis_sender is not None and self.vis_sender.enabled:
                self.env.tick_skip = 2
            self.rewards = GPURewards(new_n_envs, self.device,
                                       n_agents=new_n_agents, layout=self.layout)

            A = new_n_agents
            self._prev_actions = torch.zeros(new_n_envs, A, 8, device=self.device)
            self._controls = torch.zeros(new_n_envs, A, 8, device=self.device)
            self._current_obs = torch.zeros(self.total_agents, self.obs_size, device=self.device)

            self.env.reset_all()
            self._current_obs = build_obs_batch(self.env.state, self._prev_actions)
            self.rewards.reset_envs(
                torch.ones(new_n_envs, dtype=torch.bool, device=self.device), self.env.state)

            # Preserve policy and obs stats
            self.policy = old_policy
            self.obs_stats = old_obs_stats
            self.cumulative_timesteps = old_cumul

            if self.standardize_obs and self.obs_stats is not None:
                self.obs_stats.increment(self._current_obs, self.total_agents)

            print(f"[GPUCollector] Re-init: {new_n_envs} envs x {A} agents = "
                  f"{self.total_agents} agents/step, stage={stage}")
        else:
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
        """
        assert self.policy is not None, "Set .policy before calling collect_timesteps()"
        t0 = time.time()

        n_total = self.total_agents
        A = self.n_agents
        n_rounds = (n + n_total - 1) // n_total

        # Pre-allocate GPU buffers
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
            # 1. Normalize current obs
            obs_norm = self._normalize_obs(self._current_obs)

            # 2. Policy inference
            with torch.no_grad():
                actions_t, log_probs_t = self.policy.get_action(obs_norm)

            pre_states = obs_norm

            # 3. Parse discrete actions to continuous controls
            # actions_t: (E*A, 8) from MultiDiscrete(5,5,5,5,5,2,2,2)
            # {0,1,2,3,4} → {-1,-0.5,0,0.5,1} for first 5; {0,1} for last 3
            actions_float = actions_t.float()
            actions_reshaped = actions_float.reshape(self.n_envs, A, 8)
            self._controls[:, :, :5] = actions_reshaped[:, :, :5] * 0.5 - 1.0
            self._controls[:, :, 5:] = actions_reshaped[:, :, 5:]
            self._prev_actions.copy_(self._controls)

            # 4. Step environment
            terminals = self.env.step(self._controls)

            # 4b. Send state to visualizer (no-op when vis_sender is None)
            if self.vis_sender is not None:
                self.vis_sender.send(self.env.state)

            # 5. Compute rewards
            rewards_batch = self.rewards.compute(self.env.state, self._stage)  # (E, A)

            # 6. Build next observations
            next_obs = build_obs_batch(self.env.state, self._prev_actions)

            # 7. Update obs stats
            self._stats_step_counter += 1
            if (self.standardize_obs and
                    self._stats_step_counter >= self._steps_per_stats_increment):
                self.obs_stats.increment(next_obs, n_total)
                self._stats_step_counter = 0

            # 8. Normalize next obs
            next_norm = self._normalize_obs(next_obs)

            # 9. Record experience
            end_idx = write_idx + n_total
            all_states[write_idx:end_idx] = pre_states
            all_actions[write_idx:end_idx] = actions_float
            all_log_probs[write_idx:end_idx] = log_probs_t
            all_rewards[write_idx:end_idx] = rewards_batch.reshape(-1)
            all_next_states[write_idx:end_idx] = next_norm
            dones_expanded = terminals.unsqueeze(1).expand(-1, A).reshape(-1).float()
            all_dones[write_idx:end_idx] = dones_expanded

            write_idx = end_idx
            steps_collected += n_total

            # 10. Update current obs + reset done envs
            self._current_obs = next_obs

            if terminals.any():
                self.env.reset_done_envs(terminals)
                self.rewards.reset_envs(terminals, self.env.state)
                new_obs = build_obs_batch(self.env.state, self._prev_actions)
                reset_agents = terminals.unsqueeze(1).expand(-1, A).reshape(-1)
                self._current_obs[reset_agents] = new_obs[reset_agents]
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

        # Mark last round non-done as truncated (for GAE)
        last_round_start = actual - n_total
        if last_round_start >= 0:
            last_dones = all_dones[last_round_start:actual].reshape(self.n_envs, self.n_agents)
            last_trunc = all_truncated[last_round_start:actual].reshape(self.n_envs, self.n_agents)
            last_trunc[last_dones[:, 0] == 0] = 1.0

        self.cumulative_timesteps += actual
        elapsed = time.time() - t0

        exp = (
            all_states,
            all_actions,
            all_log_probs,
            all_rewards,
            all_next_states,
            all_dones,
            all_truncated,
        )

        return exp, [], actual, elapsed

    # Cleanup methods (no-ops)
    def stop(self): pass
    def terminate(self): pass
    def shutdown(self): pass
    def close(self): pass
    def cleanup(self): pass
