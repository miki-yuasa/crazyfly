import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Normal

from policy.layers.base import Base
from policy.layers.building_blocks import MLP


class GaussianMLPActor(Base):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        action_dim: int,
        activation: nn.Module = nn.Tanh(),
        device=torch.device("cpu"),
    ):
        super().__init__(device=device)

        self.state_dim = np.prod(input_dim)
        self.hidden_dim = hidden_dim
        self.action_dim = np.prod(action_dim)

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            self.action_dim,
            activation=activation,
            initialization="actor",
        )
        self.logstd = nn.Parameter(torch.zeros(1, self.action_dim))

        self.device = device
        self.to(self.device).to(self.dtype)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ):
        logits = self.model(state)

        ### Shape the output as desired
        mu = logits
        logstd = torch.clip(self.logstd, -5, 2)  # Clip logstd to avoid numerical issues
        std = torch.exp(logstd.expand_as(mu))
        dist = Normal(loc=mu, scale=std)

        a = dist.rsample()

        logprobs = dist.log_prob(a).unsqueeze(-1).sum(1)
        probs = torch.exp(logprobs)
        entropy = dist.entropy().sum(1)

        return a, {
            "dist": dist,
            "probs": probs,
            "logprobs": logprobs,
            "entropy": entropy,
        }

    def log_prob(self, dist: torch.distributions, actions: torch.Tensor):
        """
        Actions must be tensor
        """
        actions = actions.squeeze() if actions.shape[-1] > 1 else actions
        return dist.log_prob(actions).unsqueeze(-1).sum(1)

    def entropy(self, dist: torch.distributions):
        """
        For code consistency
        """
        return dist.entropy().unsqueeze(-1).sum(1)


class MLPCritic(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: list, activation: nn.Module = nn.Tanh()
    ):
        super().__init__()

        self.state_dim = np.prod(input_dim)

        self.model = MLP(
            self.state_dim,
            hidden_dim,
            1,
            activation=activation,
            initialization="critic",
        )

    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value
