from typing import Dict
import torch
import numpy as np

from protein_tune_rl.metrics.lm_scoring import LanguageModelScoring
from protein_tune_rl.models import create_model
from protein_tune_rl.tokenizer import create_tokenizer
from protein_tune_rl import logger


def exists(x):
    return x is not None


class IgLMScoring(LanguageModelScoring):
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer=tokenizer, pad_token='[PAD]')

    def init_tokenizer(self, tokenizer, pad_token):
        """
        Loads the tokenizer from the given model path and adds the pad token.
        """
        tokenizer = create_tokenizer(
            name="iglm_tokenizer", tokenizer_config=tokenizer, padding_side="right"
        )
        return tokenizer, tokenizer.pad_token_id

    def init_model(self, model):
        """
        Loads a causal language model from the given path and resizes the token embeddings
        to incorporate any new tokens (e.g., pad token).
        """
        try:
            model_nn = create_model(
                name="iglm",
                hf_config=model,
                vocab_size=self.tokenizer.vocab_size,
            ).to(self.device)
            model_nn.eval()
        except ValueError as e:
            raise ValueError(f"Error: Cannot load model from {model}") from e

        return model_nn

    def update_model(self, new_model):
        """
        Replace the current scoring model with a new one (e.g., the current training policy).
        """
        logger.info("Updating IGLM model in scoring function")
        self.model = new_model
        self.model.eval()

    def mask_span(self, seq, start: int, end: int, append_span: bool = False):
        """
        Mask a span in the sequence with a mask token.
        Obtained from the original IgLM implementation:
        https://github.com/Graylab/IgLM/blob/281da4fd589b71db7be8ea2670165ec9bab98667/iglm/model/utils.py#L30
        Args:
            seq (List): The original sequence.
            start (int): Start index of the span to mask.
            end (int): End index of the span to mask.
            append_span (bool): If True, append the masked span at the end.
        Returns:
            List: The masked sequence.
        """
        masked_seq = (
            seq[:start]
            + [self.tokenizer.tokenizer.mask_token]
            + seq[end:]
            + [self.tokenizer.tokenizer.sep_token]
        )
        if append_span:
            masked_seq += seq[start:end]

        return masked_seq

    def log_likelihood(
        self,
        sequence,
        chain_token,
        species_token,
        infill_range=None,
        reduction="mean",
    ):
        """
        Calculate the log-likelihood for a given sequence.
        This code is adapted from the original IgLM implementation:
        https://github.com/Graylab/IgLM/blob/281da4fd589b71db7be8ea2670165ec9bab98667/iglm/model/IgLM.py#L124

        Parameters:
            sequence (iterable): The original sequence of tokens.
            chain_token (str): The chain identifier token.
            species_token (str): The species identifier token.
            infill_range (tuple, optional): A two-element tuple (start, end) specifying the span
                                            to be masked/infilled. If provided, masking is applied
                                            and the special tokens adjusted accordingly.

        Returns:
            float: The negative cross entropy loss as the log-likelihood.
        """
        # Convert the sequence to a list (if not already a list)
        sequence = list(sequence)
        # If an infill range is given, mask out the specified span.
        if exists(infill_range):
            sequence = self.mask_span(
                sequence,
                infill_range[0],
                infill_range[1],
                append_span=True,
            )

        # Build the token sequence with chain and species tokens.
        token_seq = [chain_token, species_token] + sequence

        # Append the appropriate end token based on whether an infill is applied.
        if exists(infill_range):
            token_seq += [self.tokenizer.tokenizer.cls_token]
        else:
            token_seq += [self.tokenizer.tokenizer.sep_token]

        # Convert the token sequence into ids and move to the target device.
        token_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(token_seq)
        token_seq_tensor = torch.tensor([token_ids], dtype=torch.int).to(self.device)

        # Check for unrecognized tokens.
        assert (
            token_seq_tensor != self.tokenizer.tokenizer.unk_token_id
        ).all(), "Unrecognized token supplied in starting tokens"

        # Determine the starting point for evaluation depending on the masking.
        if exists(infill_range):
            # Find the location of the separator token id to determine eval start.
            eval_start = np.nonzero(
                token_seq_tensor[0] == self.tokenizer.tokenizer.sep_token_id
            )[0].item()
        else:
            eval_start = 1  # Skip the first token (assumed to be the starting token)

        # Obtain model logits.
        outputs = self.model(token_seq_tensor)
        logits = outputs.logits

        # Shift logits and labels so that we predict the token following each position.
        shift_logits = logits[..., eval_start:-1, :].contiguous()
        shift_labels = token_seq_tensor[..., eval_start + 1 :].contiguous().long()

        # Compute cross-entropy loss (negative log likelihood).
        nll = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction=reduction,
        )

        # Return the negative loss as the log-likelihood.
        return -nll.item()

    def __call__(self, chains: Dict):
        """
        IgLM scoring function as computed in the original IgLM paper.
        """

        infill_range = (
            len(chains["seq_pre_mask"]),
            len(chains["H"]) - len(chains["seq_post_mask"]),
        )

        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"

        return self.log_likelihood(
            chains["H"],
            chain_token,
            species_token,
            infill_range=infill_range,
            reduction="mean",
        )
