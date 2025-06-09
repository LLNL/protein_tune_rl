import copy

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from transformers.modeling_outputs import CausalLMOutput

from protein_tune_rl.util.util import compute_logp


class StateValue(nn.Module):
    def __init__(self, model_in, name):
        super(StateValue, self).__init__()
        self.model_in = copy.deepcopy(model_in.module.model)
        self.linear_head_initialized = False
        self.linear_head = nn.Linear(model_in.lm_head.in_features, 1)
        self.name = name

    def forward(
        self, input_ids, token_type_ids=None, labels=None, **kwargs
    ) -> CausalLMOutput:

        model_out = self.model_in(
            input_ids=input_ids,
            labels=labels,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )

        length = kwargs["attention_mask"].sum(1)
        last_hidden_state = model_out['hidden_states'][-1]
        output = torch.squeeze(self.linear_head(last_hidden_state), 2)
        output *= kwargs["attention_mask"]
        return output.sum(1) / length

    def save(self, path) -> None:
        self.model.save_pretrained(path)


class PPO:
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        clip: float = 0.2,
        minibatch_size: int = 2,
        entropy_weight: float = 5e-3,
        baseline: str = "mean",
        normalize_advantage: bool = False,
    ):
        self.clip = clip
        self.minibatch_size = minibatch_size
        self.entropy_weight = entropy_weight
        self.baseline = baseline
        self.normalize_adv = normalize_advantage

        self.model = model
        self.policy_optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-5)

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

        if baseline == "state_value":
            state_value = StateValue(model, "value_function")
            state_value.eval()
            state_value.to(self.device)
            if torch.cuda.is_available():                
                self.state_value = DDP(state_value, device_ids=[self.device])
            else:                
                self.state_value = DDP(state_value)
            
            self.value_optimizer = Adam(
                self.state_value.parameters(), lr=learning_rate, eps=1e-5
            )

    def _compute_clip_loss(self, adv, old_logp, state, action):
        logp = compute_logp(self.model, state, action)
        ratios = torch.exp(logp - old_logp)

        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * adv

        return -torch.min(surr1, surr2).mean()

    def step(self, reward, baseline, logp, entropy, batch):
        init_size = batch["init_size"]
        action = batch["input_ids"][:, init_size:].to(self.device).detach()
        state = {
                 "input_ids" : batch["input_ids"].to(self.device), 
                 "attention_mask" : batch["attention_mask"].to(self.device), 
                 "position_ids" : batch["position_ids"].to(self.device)
                }

        old_logp = logp.detach()

        if self.baseline == "mean":
            r_mean = self._compute_mean_across_processes(reward)
            adv = reward - r_mean
        elif self.baseline == "state_value":
            value = self.state_value(**state)
            adv = reward - value.detach()            

        if self.normalize_adv:
            if self.baseline == "mean":
                adv_mean = 0.0
            else:
                adv_mean = self._compute_mean_across_processes(adv)

            adv_var = torch.square(adv.norm(p=2)) / len(adv) - adv_mean**2
            adv_var = self._reduce_mean_across_processes(adv_var)
            adv_std = torch.sqrt(adv_var)

            adv = (adv - adv_mean) / (adv_std + 1e-10)

        if len(logp) % self.minibatch_size != 0:
            raise ValueError("Minibatch size must be a factor of the batch size")

        for start in range(0, len(logp), self.minibatch_size):
            self.policy_optimizer.zero_grad()
            if self.baseline == "state_value":
                self.value_optimizer.zero_grad()

            end = start + self.minibatch_size

            mini_state = {key: val[start:end] for key, val in state.items()}
            policy_loss = self._compute_clip_loss(
                adv[start:end], old_logp[start:end], mini_state, action[start:end]
            )
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()

            if self.baseline == "state_value":
                value = self.state_value(**mini_state)
                value_loss = nn.MSELoss()(value, reward[start:end])
                value_loss.backward()
                self.value_optimizer.step()

    def _compute_mean_across_processes(self, arg0):
        result = arg0.mean()
        return self._reduce_mean_across_processes(result)

    def _reduce_mean_across_processes(self, arg0):
        dist.all_reduce(arg0, dist.ReduceOp.SUM)
        arg0 /= dist.get_world_size()
        return arg0
