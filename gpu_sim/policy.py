"""policy.py — Custom 5-bin MultiDiscrete policy for smooth analog control.

  bins = [5,5,5,5,5,2,2,2] → 31 logits
  Continuous axes map {0,1,2,3,4} → {-1, -0.5, 0, 0.5, 1}
  Binary axes remain {0, 1}
"""

import torch
import torch.nn as nn
import numpy as np


class MultiDiscreteRolv5Bin(nn.Module):
    """5-bin version of MultiDiscreteRolv.

    Takes 31 logits, splits into 5 quintets (5 each) and 3 duets (2 each).
    Pads duets from 2→5 with -inf, then builds Categorical over (batch, 8, 5).
    """

    def __init__(self, bins):
        super().__init__()
        self.distribution = None
        self.bins = bins  # [5,5,5,5,5,2,2,2]

    def make_distribution(self, logits):
        # Split 31 logits by bins
        logits = torch.split(logits, self.bins, dim=-1)

        # First 5 groups are quintets (5 logits each) → stack to (batch, 5, 5)
        quintets = torch.stack(logits[:5], dim=-1)

        # Last 3 groups are duets (2 logits each) → pad to 5 with -inf
        duets = torch.nn.functional.pad(
            torch.stack(logits[5:], dim=-1),
            pad=(0, 0, 0, 3),  # pad last-but-one dim: 2→5
            value=float("-inf"),
        )

        # Concat → (batch, 5, 8), then swap to (batch, 8, 5)
        logits = torch.cat((quintets, duets), dim=-1).swapdims(-1, -2)

        self.distribution = torch.distributions.Categorical(logits=logits)

    def log_prob(self, action):
        return self.distribution.log_prob(action).sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


class MultiDiscreteFF5Bin(nn.Module):
    """5-bin feed-forward policy. Drop-in replacement for MultiDiscreteFF.

    Same interface: get_output(), get_action(), get_backprop_data().
    Uses bins=[5,5,5,5,5,2,2,2] = 31 output logits.
    """

    def __init__(self, input_shape, layer_sizes, device):
        super().__init__()
        self.device = device
        bins = [5, 5, 5, 5, 5, 2, 2, 2]
        n_output_nodes = sum(bins)  # 31
        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        layers = [nn.Linear(input_shape, layer_sizes[0]), nn.ReLU()]

        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_output_nodes))
        self.model = nn.Sequential(*layers).to(self.device)
        self.splits = bins
        self.multi_discrete = MultiDiscreteRolv5Bin(bins)

    def get_output(self, obs):
        t = type(obs)
        if t != torch.Tensor:
            if t != np.array:
                obs = np.asarray(obs)
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        policy_output = self.model(obs)
        return policy_output

    def get_action(self, obs, deterministic=False):
        logits = self.get_output(obs)

        if deterministic:
            start = 0
            action = []
            for split in self.splits:
                action.append(logits[..., start:start + split].argmax(dim=-1))
                start += split
            action = torch.stack(action).cpu().numpy()
            return action, 0

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.cpu(), log_prob.cpu()

    def get_backprop_data(self, obs, acts):
        logits = self.get_output(obs)

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

        entropy = distribution.entropy().to(self.device)
        log_probs = distribution.log_prob(acts).to(self.device)

        return log_probs, entropy.mean()
