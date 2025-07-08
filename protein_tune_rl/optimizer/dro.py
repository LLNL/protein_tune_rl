import torch

from protein_tune_rl import logger

torch.set_default_dtype(torch.float32)


class DRO:
    def __init__(
        self,
        policy,
        reference,
        value,
        tokenizer,
        device,
        tau,
        mean=True,
        rescaling=True,
        reward_rescaling=1.0,
    ):
        """
        Initialize the DRO (Direct Reward Optimization) training module.

        Args:
            policy (nn.Module): Trainable policy model (π_θ) producing token logits.
            reference (nn.Module): Fixed reference model (π_ref) producing token logits.
            value (nn.Module): Value network (V_φ) predicting soft value for an input prompt.
            tokenizer: Tokenizer matching both policy and reference models.
            device (torch.device): Device to perform computations on (e.g., 'cpu', 'cuda').
            tau (float): KL-temperature hyperparameter (β), controlling policy-reference divergence penalty.
            mean (bool, optional): If True, normalize losses by sequence length. Defaults to True.
            rescaling (bool, optional): If True, applies temperature rescaling on policy loss using tau. Defaults to True.
            reward_rescaling (float or callable, optional): Scaling factor (or callable transform) applied to raw rewards before loss. Defaults to 1.0.

        Attributes:
            policy, reference, value: Stored modules for forward pass.
            tokenizer, device: Tools for input processing and device allocation.
            tau: Controls strength of KL penalty in DRO objective.
            mean: Toggle for per-token loss normalization.
            rescaling: Enables or disables temperature scaling on policy loss via tau.
            reward_rescaling: Factor applied to scale rewards before usage in loss.

        Notes:
            - The policy and reference models are expected to accept input_ids and attention_mask arguments
              and output logits over the token vocabulary.
            - Loss normalization (mean=True) divides per-example losses by number of prediction tokens.
            - Reward rescaling can balance gradients when reward magnitudes vary significantly.
        """
        self.policy = policy  # -> Pi theta
        self.value = value  # -> V pi
        self.reference = reference  # -> Pi ref
        self.tokenizer = tokenizer
        self.device = device
        self.tau = tau
        self.mean = mean
        self.rescaling = rescaling
        self.reward_rescaling = reward_rescaling

    def generate_logits(self, batch, attention_mask=None):

        # Call LLM (Pi theta) and get model logits for batch prompts
        # Tensor shape (batch_size, sequence_length)
        pi_logits = self.policy(
            batch['input_ids'].to(self.device), attention_mask=attention_mask.float()
        ).logits

        # Call LLM (Pi ref) and get model logits with no gradients for batch prompts
        # Tensor shape (batch_size, sequence_length)
        with torch.no_grad():
            ref_logits = self.reference(
                batch['input_ids'].to(self.device), attention_mask=attention_mask
            ).logits

        # Tensor shape (batch_size, sequence_length-1)
        pi_logits = pi_logits[:, :-1, :]
        ref_logits = ref_logits[:, :-1, :]

        return pi_logits, ref_logits

    def calculate_loss(self, batch):
        # Tensor shape (batch_size, sequence_length)
        labels = batch['labels'].clone().to(self.device)
        # Create attention mask so only completion tokens are attended to
        # Tensor shape (batch_size, sequence_length)
        policy_attention_mask = torch.ones(labels.shape).to(self.device) * (labels != 0)
        # Remove first token -> Tensor shape (batch_size, sequence_length-1)
        labels = labels[:, 1:].clone()

        # Tensor shape (batch_size, sequence_length-1)
        # Create loss mask where all values are zero except indices of completion tokens
        # Loss mask is used to compute loss only over completion tokens
        loss_mask = (labels != -100) & (labels != 0)
        labels[labels == -100] = 0

        # Tensor shape (batch_size, 1)
        rewards = batch['rewards'].to(self.device).unsqueeze(1).float().flatten()

        # Rescale rewards if rescaling is enabled
        rewards = rewards * self.reward_rescaling

        # Tensor shape (batch_size, sequence_length-1, vocab_size)
        pi_logits, ref_logits = self.generate_logits(
            batch, attention_mask=policy_attention_mask
        )

        # Check shapes and handle error
        if pi_logits.shape[:-1] != labels.shape:
            logger.error("Error : Logits and labels are not the same shape")

        # Tensor shape (batch_size, sequence_length-1)
        # Get log probability for tokens of a given completion over the batch
        pi_log_probs = torch.gather(
            pi_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        ref_log_probs = torch.gather(
            ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)
        # log_ratio -> log(pi) - log(pi_ref) -> log (pi / pi_ref)
        log_ratio = pi_log_probs - ref_log_probs

        # Tensor shape (batch_size, 1)
        # Call V pi for given prompts
        value_attention_mask = torch.ones(batch['prompts'].shape) * (
            batch['prompts'] != 0
        )
        value = (
            self.value(
                batch['prompts'].to(self.device),
                attention_mask=value_attention_mask.to(self.device),
            )
            .float()
            .flatten()
        )

        # Detach from graph to remove gradient calculation for loss functions
        value_no_grad = value.clone().detach()
        log_ratio_no_grad = log_ratio.clone().detach()

        # DRO-V algorithm 1 https://arxiv.org/pdf/2405.19107
        # Policy and value loss
        loss_denom = loss_mask.sum(-1) if self.mean else 1.0
        policy_tau = 1.0 if self.rescaling else self.tau

        policy_loss = (
            -policy_tau
            * (
                ((pi_log_probs * loss_mask).sum(-1) / loss_denom)
                * (rewards - value_no_grad)
                - policy_tau
                / 2
                * torch.pow((log_ratio * loss_mask).sum(-1) / loss_denom, 2)
            ).mean()
        )

        value_loss = (
            (
                value_no_grad
                - rewards
                + self.tau * ((log_ratio_no_grad * loss_mask).sum(-1) / loss_denom)
            )
            * value
        ).mean()

        return policy_loss, value_loss
