import re
import torch
from torch.nn.utils.rnn import pad_sequence


class DPOCollator:
    """Collator for Direct Preference Optimization (DPO) training batches."""

    def __init__(self, tokenizer, eval=False):
        """
        Args:
            tokenizer: Tokenizer for IgLM (callable on a spaced string to produce token IDs).
            eval (bool): If True, prepare batch for evaluation (prompts only). Otherwise, prepare training pairs.
        """
        self.tokenizer = tokenizer
        self.eval = eval

    def __call__(self, batch):
        """
        Collate a batch of raw examples into model inputs for DPO training.

        Each raw example should contain:
            - 'prompt': heavy chain sequence with a [MASK] token indicating the infill region.
            - 'completion_pos': the preferred completion sequence (string of amino acids).
            - 'completion_neg': the rejected completion sequence.
            - 'LC': (optional) light chain sequence, which is not directly used in input concatenation for IgLM.

        Returns:
            dict with keys:
            - input_ids_pos: Tensor of token IDs for [HEAVY] [HUMAN] <masked_prompt> [SEP] <pos_completion> [CLS].
            - input_ids_neg: Tensor of token IDs for [HEAVY] [HUMAN] <masked_prompt> [SEP] <neg_completion> [CLS].
            - labels_pos: Tensor of same shape as input_ids_pos, with -100 for non-completion tokens and token IDs for pos completion tokens.
            - labels_neg: Tensor of same shape as input_ids_neg, with -100 for non-completion tokens and token IDs for neg completion tokens.
            - __row_idx__: Tensor of original indices for reference (if provided in batch).

            (If eval=True, this collator could return only prompt-related inputs, but by default eval mode is not used for DPO.)
        """
        if self.eval:
            # For evaluation, we might only need to prepare the prompt (and possibly LC context).
            # (DPO evaluation usage may vary; here we simply return the prompt tokenization.)
            prompts = []
            for item in batch["prompt"]:
                # Prepare prompt with heavy chain context only
                # Ensure correct spacing for amino acids and special tokens
                masked_seq = re.findall(r'\[MASK\]|.', item)
                spaced_prompt = "[HEAVY] [HUMAN] " + " ".join(masked_seq) + " [SEP]"
                prompts.append(spaced_prompt)
            tokenized_prompts = list(map(self.tokenizer, prompts))
            input_ids = pad_sequence(
                tokenized_prompts, batch_first=True, padding_value=0
            )
            output = {
                "__row_idx__": batch.get("__row_idx__", None),
                "input_ids": input_ids,
            }
            if "LC" in batch:
                output["LC"] = batch["LC"]
            return output

        # Prepare lists to collect tokenized sequences and label masks
        tokenized_pos_list = []
        tokenized_neg_list = []
        labels_pos_list = []
        labels_neg_list = []
        row_idx_list = []
        lc_list = []

        # Get all keys in batch
        batch_keys = batch.keys()
        batch_size = len(batch["prompt"])  # assuming all lists are same length

        for i in range(batch_size):
            # Build item as a dict with one entry per key
            item = {
                key: batch[key][i] for key in batch_keys if isinstance(batch[key], list)
            }

            heavy_masked = item["prompt"]  # heavy chain with [MASK] in place
            pos_completion = item["completion_pos"]
            neg_completion = item["completion_neg"]

            # Space out the heavy sequence (each amino acid) and keep [MASK] as a single token
            # re.findall ensures '[MASK]' stays intact, and each other character becomes a token.
            masked_tokens = re.findall(r'\[MASK\]|.', heavy_masked)
            spaced_masked_prompt = (
                "[HEAVY] [HUMAN] " + " ".join(masked_tokens) + " [SEP]"
            )

            # Space out completion sequences (each char) and append [CLS] end token
            spaced_pos_comp = " ".join(pos_completion) + " [CLS]"
            spaced_neg_comp = " ".join(neg_completion) + " [CLS]"

            # Tokenize prompt and completions separately
            prompt_ids = self.tokenizer(
                spaced_masked_prompt
            )  # tensor of prompt token IDs (including [SEP])
            pos_comp_ids = self.tokenizer(
                spaced_pos_comp
            )  # tensor of pos completion token IDs (incl. [CLS])
            neg_comp_ids = self.tokenizer(
                spaced_neg_comp
            )  # tensor of neg completion token IDs (incl. [CLS])

            # Concatenate prompt and completion token sequences to form full input sequences
            # (We simply join tensors; prompt ends with [SEP], completion includes [CLS] at end)
            combined_pos_ids = torch.cat([prompt_ids, pos_comp_ids])
            combined_neg_ids = torch.cat([prompt_ids, neg_comp_ids])

            # Create label masks for positive and negative sequences.
            # We mark positions of the completion tokens (including the final [CLS]) with their token IDs, and others as -100.
            # Length of combined sequence minus 1 (we include the last [CLS] token from prediction)
            mask_len_pos = combined_pos_ids.size(0)
            mask_len_neg = combined_neg_ids.size(0)

            # Initialize label arrays with -100 for all positions that will not be predicted
            labels_pos = [-100] * mask_len_pos
            labels_neg = [-100] * mask_len_neg

            def fill_labels(labels, comp_ids):
                """Align the completion token IDs (including [CLS]) to the end of the label tensor."""
                n_tokens = comp_ids.size(0)
                if n_tokens > 0:
                    labels[-n_tokens:] = comp_ids.tolist()

            # Fill in the completion token IDs into the label masks
            fill_labels(labels_pos, pos_comp_ids)
            fill_labels(labels_neg, neg_comp_ids)

            # Convert to tensors
            labels_pos_tensor = torch.tensor(labels_pos, dtype=torch.long)
            labels_neg_tensor = torch.tensor(labels_neg, dtype=torch.long)

            # Collect tokens and labels
            tokenized_pos_list.append(combined_pos_ids)
            tokenized_neg_list.append(combined_neg_ids)
            labels_pos_list.append(labels_pos_tensor)
            labels_neg_list.append(labels_neg_tensor)

            # Append row index and LC if available
            if "__row_idx__" in item:
                row_idx_list.append(item["__row_idx__"])
            if "LC" in item:
                lc_list.append(item["LC"])

        # Pad all sequences in the batch to the same length
        input_ids_pos = pad_sequence(
            tokenized_pos_list, batch_first=True, padding_value=0
        )
        input_ids_neg = pad_sequence(
            tokenized_neg_list, batch_first=True, padding_value=0
        )
        labels_pos = pad_sequence(
            labels_pos_list, batch_first=True, padding_value=0
        )  # pad with 0 (will be ignored as no-loss)
        labels_neg = pad_sequence(labels_neg_list, batch_first=True, padding_value=0)

        # Prepare output batch dict
        output_batch = {
            "input_ids_pos": input_ids_pos,
            "input_ids_neg": input_ids_neg,
            "labels_pos": labels_pos,
            "labels_neg": labels_neg,
        }
        # Include row indices and LC if present (for potential debugging/analysis)
        if row_idx_list:
            output_batch["__row_idx__"] = torch.tensor(row_idx_list, dtype=torch.long)
        if lc_list:
            output_batch["LC"] = lc_list

        return output_batch
