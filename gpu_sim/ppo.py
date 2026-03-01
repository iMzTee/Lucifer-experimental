"""ppo.py — Self-contained PPO implementation for LuciferBot.

Provides ValueEstimator, ExperienceBuffer, and PPOLearner with AMP
mixed-precision training. No external RL library dependencies.

Checkpoint-compatible with existing PPO_POLICY.pt / PPO_VALUE_NET.pt files.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

from gpu_sim.policy import MultiDiscreteFF5Bin


class ValueEstimator(nn.Module):
    """Critic network: maps observations to scalar state-value estimates."""

    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device
        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        layers.append(nn.Linear(layer_sizes[-1], 1))
        self.model = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        t = type(x)
        if t != torch.Tensor:
            if t != np.array:
                x = np.asarray(x)
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return self.model(x)


class ExperienceBuffer:
    """GPU-resident experience buffer with FIFO overflow trimming."""

    def __init__(self, max_size, device, seed=123):
        self.device = device
        self.seed = seed
        self.max_size = max_size
        self.states = torch.FloatTensor().to(self.device)
        self.actions = torch.FloatTensor().to(self.device)
        self.log_probs = torch.FloatTensor().to(self.device)
        self.rewards = torch.FloatTensor().to(self.device)
        self.next_states = torch.FloatTensor().to(self.device)
        self.dones = torch.FloatTensor().to(self.device)
        self.truncated = torch.FloatTensor().to(self.device)
        self.values = torch.FloatTensor().to(self.device)
        self.advantages = torch.FloatTensor().to(self.device)

    @staticmethod
    def _cat(old, new, max_size):
        combined = torch.cat((old, new), dim=0)
        if combined.shape[0] > max_size:
            combined = combined[-max_size:]
        return combined

    def submit_experience(self, states, actions, log_probs, rewards,
                          next_states, dones, truncated, values, advantages):
        _cat = ExperienceBuffer._cat
        ms = self.max_size
        dev = self.device
        self.states = _cat(self.states, torch.as_tensor(states, dtype=torch.float32, device=dev), ms)
        self.actions = _cat(self.actions, torch.as_tensor(actions, dtype=torch.float32, device=dev), ms)
        self.log_probs = _cat(self.log_probs, torch.as_tensor(log_probs, dtype=torch.float32, device=dev), ms)
        self.rewards = _cat(self.rewards, torch.as_tensor(rewards, dtype=torch.float32, device=dev), ms)
        self.next_states = _cat(self.next_states, torch.as_tensor(next_states, dtype=torch.float32, device=dev), ms)
        self.dones = _cat(self.dones, torch.as_tensor(dones, dtype=torch.float32, device=dev), ms)
        self.truncated = _cat(self.truncated, torch.as_tensor(truncated, dtype=torch.float32, device=dev), ms)
        self.values = _cat(self.values, torch.as_tensor(values, dtype=torch.float32, device=dev), ms)
        self.advantages = _cat(self.advantages, torch.as_tensor(advantages, dtype=torch.float32, device=dev), ms)

    def _get_samples(self, indices):
        return (self.actions[indices],
                self.log_probs[indices],
                self.states[indices],
                self.values[indices],
                self.advantages[indices])

    def get_all_batches_shuffled(self, batch_size):
        """GPU-native batch shuffling using torch.randperm (no numpy)."""
        total_samples = self.rewards.shape[0]
        indices = torch.randperm(total_samples, device=self.rewards.device)
        start_idx = 0
        while start_idx + batch_size <= total_samples:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def clear(self):
        del self.states, self.actions, self.log_probs, self.rewards
        del self.next_states, self.dones, self.truncated, self.values, self.advantages
        self.__init__(self.max_size, self.device, self.seed)


class PPOLearner:
    """PPO trainer with AMP mixed-precision and gradient scaling."""

    def __init__(self, obs_space_size, act_space_size, device, batch_size,
                 mini_batch_size, n_epochs, policy_layer_sizes, critic_layer_sizes,
                 policy_lr, critic_lr, clip_range, ent_coef,
                 # Ignored — kept for migration convenience
                 policy_type=None, continuous_var_range=None):
        self.device = device
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.cumulative_model_updates = 0

        self.policy = MultiDiscreteFF5Bin(obs_space_size, policy_layer_sizes, device)
        self.value_net = ValueEstimator(obs_space_size, critic_layer_sizes, device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=critic_lr)
        self.value_loss_fn = nn.MSELoss()
        self._grad_scaler = torch.amp.GradScaler('cuda')

    def learn(self, exp):
        """PPO update with AMP autocast and gradient scaling."""
        scaler = self._grad_scaler

        n_iterations = 0
        n_minibatch_iterations = 0
        mean_entropy = 0
        mean_divergence = 0
        mean_val_loss = 0
        clip_fractions = []

        t1 = time.time()
        for epoch in range(self.n_epochs):
            for batch in exp.get_all_batches_shuffled(self.batch_size):
                (batch_acts, batch_old_probs, batch_obs,
                 batch_target_values, batch_advantages) = batch
                batch_acts = batch_acts.view(self.batch_size, -1)
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                for minibatch_slice in range(0, self.batch_size, self.mini_batch_size):
                    start = minibatch_slice
                    stop = start + self.mini_batch_size
                    acts = batch_acts[start:stop].to(self.device)
                    obs = batch_obs[start:stop].to(self.device)
                    advantages = batch_advantages[start:stop].to(self.device)
                    old_probs = batch_old_probs[start:stop].to(self.device)
                    target_values = batch_target_values[start:stop].to(self.device)

                    with torch.amp.autocast('cuda'):
                        vals = self.value_net(obs).view_as(target_values)
                        log_probs, entropy = self.policy.get_backprop_data(obs, acts)
                        log_probs = log_probs.view_as(old_probs)
                        ratio = torch.exp(log_probs - old_probs)
                        clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                        minibatch_ratio = self.mini_batch_size / self.batch_size
                        value_loss = self.value_loss_fn(vals, target_values) * minibatch_ratio
                        ppo_loss = (policy_loss - entropy * self.ent_coef) * minibatch_ratio

                    with torch.no_grad():
                        log_ratio = log_probs.float() - old_probs.float()
                        kl = (torch.exp(log_ratio) - 1) - log_ratio
                        kl = kl.mean().detach().cpu().item()
                        clip_fraction = torch.mean(
                            (torch.abs(ratio.float() - 1) > self.clip_range).float()).cpu().item()
                        clip_fractions.append(clip_fraction)

                    scaler.scale(ppo_loss).backward()
                    scaler.scale(value_loss).backward()
                    mean_val_loss += (value_loss / minibatch_ratio).cpu().detach().item()
                    mean_divergence += kl
                    mean_entropy += entropy.cpu().detach().item()
                    n_minibatch_iterations += 1

                scaler.unscale_(self.policy_optimizer)
                scaler.unscale_(self.value_optimizer)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                scaler.step(self.policy_optimizer)
                scaler.step(self.value_optimizer)
                scaler.update()
                n_iterations += 1

        if n_iterations == 0: n_iterations = 1
        if n_minibatch_iterations == 0: n_minibatch_iterations = 1

        mean_entropy /= n_minibatch_iterations
        mean_divergence /= n_minibatch_iterations
        mean_val_loss /= n_minibatch_iterations
        mean_clip = np.mean(clip_fractions) if clip_fractions else 0

        self.cumulative_model_updates += n_iterations

        return {
            "PPO Batch Consumption Time": (time.time() - t1) / n_iterations,
            "Cumulative Model Updates": self.cumulative_model_updates,
            "Policy Entropy": mean_entropy,
            "Mean KL Divergence": mean_divergence,
            "Value Function Loss": mean_val_loss,
            "SB3 Clip Fraction": mean_clip,
        }

    def save_to(self, folder_path):
        os.makedirs(folder_path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(folder_path, "PPO_POLICY.pt"))
        torch.save(self.value_net.state_dict(), os.path.join(folder_path, "PPO_VALUE_NET.pt"))
        torch.save(self.policy_optimizer.state_dict(), os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"))
        torch.save(self.value_optimizer.state_dict(), os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"))

    def load_from(self, folder_path):
        assert os.path.exists(folder_path), f"PPO LEARNER CANNOT FIND FOLDER {folder_path}"
        self.policy.load_state_dict(torch.load(os.path.join(folder_path, "PPO_POLICY.pt"), weights_only=True))
        self.value_net.load_state_dict(torch.load(os.path.join(folder_path, "PPO_VALUE_NET.pt"), weights_only=True))
        self.policy_optimizer.load_state_dict(torch.load(os.path.join(folder_path, "PPO_POLICY_OPTIMIZER.pt"), weights_only=True))
        self.value_optimizer.load_state_dict(torch.load(os.path.join(folder_path, "PPO_VALUE_NET_OPTIMIZER.pt"), weights_only=True))
