import torch
import torch.nn.functional as F  # (we'll use it later for a numerically-stable loss)

torch.autograd.set_detect_anomaly(True)


def _pad_to_len(x: torch.Tensor, target_len: int, pad_id: int = 0) -> torch.Tensor:
    # x: [B, L]; right-pad to target_len with pad_id
    pad = target_len - x.size(1)
    return x if pad <= 0 else F.pad(x, (0, pad), value=pad_id)


class DPO:
    def __init__(
        self,
        policy,
        reference,
        tokenizer,
        device,
        beta=0.1,
        length_normalize: bool = False,
    ):
        """
        Direct Preference Optimization (DPO) optimizer.

        Args:
            policy (nn.Module): The trainable policy model (π_θ).
            reference (nn.Module): The frozen reference model (π_ref).
            tokenizer: Tokenizer for the model (to obtain vocab size, special tokens, etc.).
            device (torch.device): Device to run computations on.
            beta (float): Scaling factor β for the DPO loss.
            length_normalize (bool): If True, normalize log-probs by sequence length.
        """
        self.policy = policy
        self.reference = reference
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta
        self.length_normalize = length_normalize

        # Grab the pad token ID (default to 0 if not set)
        self.PAD = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

    def calculate_loss(self, batch):
        # ---- Move, clone labels (we’ll edit -100 -> 0) ----
        input_ids_pos = batch["input_ids_pos"].to(self.device)
        input_ids_neg = batch["input_ids_neg"].to(self.device)
        labels_pos = batch["labels_pos"].clone().to(self.device)
        labels_neg = batch["labels_neg"].clone().to(self.device)

        B = input_ids_pos.size(0)
        # 1) Pad inputs to a common sequence length L
        L = max(input_ids_pos.size(1), input_ids_neg.size(1))
        input_ids_pos = _pad_to_len(input_ids_pos, L, pad_id=self.PAD)
        input_ids_neg = _pad_to_len(input_ids_neg, L, pad_id=self.PAD)

        # 2) Build attention masks and concat pos/neg -> one batch
        att_pos = (input_ids_pos != self.PAD).to(self.device).float()
        att_neg = (input_ids_neg != self.PAD).to(self.device).float()
        input_ids_cat = torch.cat([input_ids_pos, input_ids_neg], dim=0)  # [2B, L]
        att_cat = torch.cat([att_pos, att_neg], dim=0)  # [2B, L]

        # 3) POLICY forward ONCE (training graph)
        #    (pass use_cache=False if your model supports it)
        policy_out = self.policy(input_ids_cat, attention_mask=att_cat, use_cache=False)
        policy_logits = policy_out.logits  # [2B, L, V]
        policy_logits_pos = policy_logits[:B]
        policy_logits_neg = policy_logits[B:]

        # 4) REFERENCE forward ONCE under inference_mode (frozen)
        base_ref = (
            self.reference.module
            if hasattr(self.reference, "module")
            else self.reference
        )
        with torch.inference_mode():  # avoids version counter bumps
            ref_out = base_ref(input_ids_cat, attention_mask=att_cat, use_cache=False)
            ref_logits = ref_out.logits  # [2B, L, V]
            ref_logits_pos = ref_logits[:B]
            ref_logits_neg = ref_logits[B:]

        # 5) Next-token alignment: drop last logit, drop first label (do this ONCE)
        policy_logits_pos = policy_logits_pos[:, :-1, :].contiguous()
        policy_logits_neg = policy_logits_neg[:, :-1, :].contiguous()
        ref_logits_pos = ref_logits_pos[:, :-1, :].contiguous()
        ref_logits_neg = ref_logits_neg[:, :-1, :].contiguous()

        # Pad labels to L, then drop first label -> length L-1 (matches logits)
        labels_pos = _pad_to_len(labels_pos, L, pad_id=self.PAD)[:, 1:].contiguous()
        labels_neg = _pad_to_len(labels_neg, L, pad_id=self.PAD)[:, 1:].contiguous()

        # 6) Build loss masks over completion tokens only
        mask_pos = (labels_pos != -100) & (labels_pos != self.PAD)
        mask_neg = (labels_neg != -100) & (labels_neg != self.PAD)

        # Replace -100 with 0 to make indices safe for gather
        labels_pos[labels_pos == -100] = self.PAD
        labels_neg[labels_neg == -100] = self.PAD

        # 7) Gather log-probs and sum over masked positions
        def seq_logprob(logits, labels, mask):
            # logits: [B, T, V], labels: [B, T], mask: [B, T] (bool)
            lp = torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(2)).squeeze(
                2
            )  # [B, T]
            m = mask.float()
            token_sum = (lp * m).sum(dim=1)  # [B]
            if self.length_normalize:
                lengths = m.sum(dim=1).clamp_min(1.0)  # avoid div-by-zero
                return token_sum / lengths
            else:
                return token_sum

        pi_pos = seq_logprob(policy_logits_pos, labels_pos, mask_pos)  # log πθ(y+|x)
        pi_neg = seq_logprob(policy_logits_neg, labels_neg, mask_neg)  # log πθ(y-|x)
        ref_pos = seq_logprob(ref_logits_pos, labels_pos, mask_pos)  # log πref(y+|x)
        ref_neg = seq_logprob(ref_logits_neg, labels_neg, mask_neg)  # log πref(y-|x)

        # 8) DPO margin and loss:  L = - E[ log σ(β * ( (log πθ(y+|x) - log πref(y+|x)) - (log πθ(y-|x) - log πref(y-|x)) )) ]
        diff = (pi_pos - ref_pos) - (pi_neg - ref_neg)  # [B]
        # diff is equivalent to : (pi_pos - pi_neg) - (ref_pos - ref_neg)

        # F.softplus(x) = log(1 + exp(x)) is a stable way to compute -log σ(-x)
        policy_loss = F.softplus(-self.beta * diff).mean()  # == -log σ(β*diff)

        # Interpretation: we want pi_pos - pi_neg to be larger than ref_pos - ref_neg by a margin of 1/beta.
        # Typically, ref_pos - ref_neg is close to 0, so we want pi_pos - pi_neg to be positive and large.
        # (the smaller the beta, the larger the margin 1/beta)

        return (
            policy_loss,
            diff.detach(),
            pi_pos.detach(),
            pi_neg.detach(),
            ref_pos.detach(),
            ref_neg.detach(),
        )
