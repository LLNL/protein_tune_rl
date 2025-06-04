import re

import torch
from torch.nn.utils.rnn import pad_sequence


class DROCollator:
    def __init__(
        self,
        model_name,
        tokenizer,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer

    def create_mask(self, tokenized_sequences, tokenized_completions):
        batch_mask = []
        for sequence, completion in zip(tokenized_sequences, tokenized_completions):

            # create mask to compute loss only on completion tokens
            # use -100 as to not infere with attention mask creation
            mask = [-100 for _ in sequence[:-1]]
            mask[-len(completion) + 1 :] = list(completion)

            batch_mask.append(torch.tensor(mask))

        return batch_mask

    def _pad_ragged_tensors(self, batch_tensors):
        length_of_first = batch_tensors[0].size(0)
        are_tensors_same_length = all(
            x.size(0) == length_of_first for x in batch_tensors
        )
        if are_tensors_same_length:
            return torch.stack(batch_tensors, dim=0)
        else:
            return pad_sequence(batch_tensors, batch_first=True)

    def _batch_tokenize(self, sequences, completion):
        tokenized_sequences = list(map(self.tokenizer, sequences))
        tokenized_completions = list(map(self.tokenizer, completion))

        mask = self.create_mask(tokenized_sequences, tokenized_completions)

        input_sequences = self._pad_ragged_tensors(tokenized_sequences)
        input_completions = self._pad_ragged_tensors(tokenized_completions)
        input_mask = self._pad_ragged_tensors(mask)
        return input_sequences, input_completions, input_mask

    def __call__(self, batch, eval=False):

        masked_prompts, masked_prompts_with_completions, spaced_completions, sequences_pre_mask, sequences_post_mask = (
            [],
            [],
            [],
            [],
            [],
        )
        # NOTE: In cases where a given completion pattern occurs in multiple different spans for a given prompt
        # this code will insert multiple masks. This code should be changed to handle such scenarios in the future.
        for prompt, completion in zip(batch["prompts"], batch["completions"]):
            masked_prompt = ' '.join(prompt).replace(
                ' '.join(str(completion)), "[MASK]"
            )
            spaced_completions.append(" ".join(completion) + "[CLS]")

            masked_region_idx = re.search(completion, prompt)
            seq_pre_mask = prompt[: masked_region_idx.start()]
            seq_post_mask = prompt[masked_region_idx.end() :]

            prompt = (
                "[HEAVY]"
                + " "
                + "[HUMAN]"
                + " "
                + masked_prompt
                + " "
                + "[SEP]"
                + " "
                + " ".join(completion)
                + " "
                + "[CLS]"
            )

            if eval:
                prompt = (
                    '[HEAVY]' + " " + "[HUMAN]" + " " + masked_prompt + " " + "[SEP]"
                )

            masked_prompts_with_completions.append(prompt)
            masked_prompts.append(
                "[HEAVY]" + " " + "[HUMAN]" + " " + masked_prompt + " " + "[SEP]"
            )

            sequences_pre_mask.append(seq_pre_mask)
            sequences_post_mask.append(seq_post_mask)

        (
            tokenized_masked_prompts_with_completions,
            __,
            input_mask,
        ) = self._batch_tokenize(masked_prompts_with_completions, spaced_completions)
        tokenized_masked_prompts, __, __ = self._batch_tokenize(
            masked_prompts, spaced_completions
        )

        if eval:
            return {
                "input_ids": tokenized_masked_prompts_with_completions,
                "prompts": tokenized_masked_prompts,
                "labels": input_mask,
                "LC": batch["LC"],
                "seq_pre_mask" : sequences_pre_mask,
                "seq_post_mask" : sequences_post_mask
            }

        return {
            "input_ids": tokenized_masked_prompts_with_completions,
            "prompts": tokenized_masked_prompts,
            "labels": input_mask,
            "rewards": batch["rewards"],
            "seq_pre_mask" : sequences_pre_mask,
            "seq_post_mask" : sequences_post_mask
        }
