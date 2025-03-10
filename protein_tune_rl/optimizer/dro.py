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
    ):
        self.policy = policy  # -> Pi theta
        self.value = value  # -> V pi
        self.reference = reference  # -> Pi ref
        self.tokenizer = tokenizer
        self.device = device
        self.tau = tau
        self.mean = mean
        self.rescaling = rescaling

    def generate_logits(self, batch, attention_mask=None):

        # Call LLM (Pi theta) and get model logits for batch prompts
        # Tensor shape (batch_size, sequence_length)
        pi_logits = self.policy(
            batch['input_ids'].to(self.device), attention_mask
        ).logits

        # Call LLM (Pi ref) and get model logits with no gradients for batch prompts
        # Tensor shape (batch_size, sequence_length)
        with torch.no_grad():
            ref_logits = self.reference(
                batch['input_ids'].to(self.device), attention_mask
            ).logits

        # Tensor shape (batch_size, sequence_length-1)
        pi_logits = pi_logits[:, :-1, :]
        ref_logits = ref_logits[:, :-1, :]

        return pi_logits, ref_logits

    def calculate_loss(self, batch):
        # Tensor shape (batch_size, sequence_length)
        labels = batch['labels'].clone().to(self.device)
        # Tensor shape (batch_size, sequence_length-1)
        labels = labels[:, 1:].clone()

        # Tensor shape (batch_size, sequence_length-1)
        # Create loss mask where all values are zero except indices of completion tokens
        # Loss mask is used to compute loss only over completion tokens
        loss_mask = (labels != -100) & (labels != 0)

        # Create attention mask so only completion tokens are attended to
        # Tensor shape (batch_size, sequence_length-1)
        attention_mask = torch.ones(labels.shape).to(self.device) * (labels != 0)
        labels[labels == -100] = 0

        # Tensor shape (batch_size, 1)
        rewards = batch['rewards'].to(self.device).unsqueeze(1).float().flatten()

        # Tensor shape (batch_size, sequence_length-1, vocab_size)
        pi_logits, ref_logits = self.generate_logits(batch, attention_mask)

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
        value = self.value(batch['prompts'].to(self.device)).float().flatten()

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
