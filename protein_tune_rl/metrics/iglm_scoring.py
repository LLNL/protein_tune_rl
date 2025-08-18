from typing import Dict
import torch
import numpy as np

from protein_tune_rl.metrics.lm_scoring import LanguageModelScoring
from protein_tune_rl.models import create_model
from protein_tune_rl.tokenizer import create_tokenizer
from protein_tune_rl import logger


def get_seq_length(seq):
    """
    Some of the sequence preparation functions return a list of sequences,
    e.g. chains["seq_pre_mask"] = ['EVQLVESGGGLVQP ... TAVYYCAR']
    This function returns the length of the first sequence in the list.
    If the input is a single sequence, it returns its length.
    """
    if isinstance(seq, list):
        return len(seq[0]) if seq else 0
    else:
        return len(seq)


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

    def _encode_with_infill(self, sequence, chain_token, species_token, infill_range):
        """
        Prepares and encodes a token sequence for infill evaluation.

        This method constructs the input token sequence by prepending chain and species tokens,
        optionally masking a span if infill_range is provided, and appending the appropriate end token.
        It then converts the tokens to IDs, creates a tensor on the target device, and determines
        the evaluation start index for infill scoring.

        Args:
            sequence (iterable): The original sequence of tokens.
            chain_token (str): The chain identifier token.
            species_token (str): The species identifier token.
            infill_range (tuple, optional): A two-element tuple (start, end) specifying the span
                                            to be masked/infilled. If provided, masking is applied.

        Returns:
            token_seq_tensor (torch.Tensor): The encoded token sequence tensor.
            eval_start (int): The index in the sequence where evaluation should begin.
        """
        token_seq = self._prepare_token_sequence_with_infill(
            sequence, infill_range, chain_token, species_token
        )
        token_seq += (
            [self.tokenizer.tokenizer.cls_token]
            if exists(infill_range)
            else [self.tokenizer.tokenizer.sep_token]
        )

        token_ids = self.tokenizer.tokenizer.convert_tokens_to_ids(token_seq)
        token_seq_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=self.device
        )

        assert (
            token_seq_tensor != self.tokenizer.tokenizer.unk_token_id
        ).all(), "Unrecognized token supplied in starting tokens"

        if exists(infill_range):
            eval_start = (
                (token_seq_tensor[0] == self.tokenizer.tokenizer.sep_token_id)
                .nonzero()[0]
                .item()
            )
        else:
            eval_start = 1
        return token_seq_tensor, eval_start

    def token_logits_over_infill(
        self, sequence, chain_token, species_token, infill_range
    ):
        """
        Returns the model's logits for each token position over the infill span.

        This method encodes the input sequence with chain and species tokens, applies infill masking if specified,
        and passes the sequence through the model. It then extracts the logits corresponding to the infill region.

        Args:
            sequence (iterable): The original sequence of tokens.
            chain_token (str): The chain identifier token.
            species_token (str): The species identifier token.
            infill_range (tuple): A two-element tuple (start, end) specifying the span to be masked/infilled.

        Returns:
            torch.Tensor: Logits for each token position in the infill region, shape [L, V],
                        where L is the infill length and V is the vocabulary size.
        """
        token_seq_tensor, eval_start = self._encode_with_infill(
            sequence, chain_token, species_token, infill_range
        )
        with torch.no_grad():
            outputs = self.model(token_seq_tensor)
            logits = outputs.logits
        shift_logits = logits[:, eval_start:-1, :].contiguous()  # [1, L, V]
        return shift_logits.squeeze(0)  # [L, V]

    def distribution_over_infill(
        self, sequence, chain_token, species_token, infill_range
    ):
        """
        Returns the probability distribution over the vocabulary for each token position in the infill span.

        This method encodes the input sequence with chain and species tokens, applies infill masking if specified,
        and passes the sequence through the model. It then extracts the logits for the infill region and applies
        softmax to obtain probabilities.

        Args:
            sequence (iterable): The original sequence of tokens.
            chain_token (str): The chain identifier token.
            species_token (str): The species identifier token.
            infill_range (tuple): A two-element tuple (start, end) specifying the span to be masked/infilled.

        Returns:
            torch.Tensor: Probability distribution for each token position in the infill region, shape [L, V],
                        where L is the infill length and V is the vocabulary size.
        """
        logits = self.token_logits_over_infill(
            sequence, chain_token, species_token, infill_range
        )
        return torch.nn.functional.softmax(logits.float(), dim=-1)

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
        token_seq = self._prepare_token_sequence_with_infill(
            sequence, infill_range, chain_token, species_token
        )
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

    def _prepare_token_sequence_with_infill(
        self, sequence, infill_range, chain_token, species_token
    ):
        sequence = list(sequence)
        if exists(infill_range):
            sequence = self.mask_span(
                sequence, infill_range[0], infill_range[1], append_span=True
            )
        return [chain_token, species_token] + sequence

    def __call__(self, chains: Dict):
        """
        IgLM scoring function as computed in the original IgLM paper.
        """

        chains_seq_pre_mask_length = get_seq_length(chains["seq_pre_mask"])
        chains_seq_post_mask_length = get_seq_length(chains["seq_post_mask"])
        chains_H_length = get_seq_length(chains["H"])

        infill_range = (
            chains_seq_pre_mask_length,
            chains_H_length - chains_seq_post_mask_length,
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
