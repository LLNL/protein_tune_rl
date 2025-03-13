import torch
from torch.optim import Adam


class Reinforce:
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        entropy_weight: float = 5e-3,
    ):
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.entropy_weight = entropy_weight

    def _compute_loss(self, reward, baseline, logp, entropy):
        return (
            -((reward - baseline) * logp).mean() - self.entropy_weight * entropy.mean()
        )

    def step(self, reward, baseline, logp, entropy, sequences):
        self.optimizer.zero_grad()
        loss = self._compute_loss(reward, baseline, logp, entropy)
        loss.backward()
        self.optimizer.step()
