import re

import torch
from torch.nn.utils.rnn import pad_sequence


class DROCollator:
    """Collator for DRO training or evaluation of IgLM."""

    def __init__(self, tokenizer, eval=False):
        self.tokenizer = tokenizer
        self.eval = eval

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

    def __call__(self, batch):
        """Collate raw sequence data for DRO training or evaluation of IgLM.
        Following the IgLM paper, sequences in the batch are masked for infilling.
        The batch of collated sequences is padded on the right.

        For example, suppose the batch of raw sequences is:
            [
                'EVQLVESIQP',
                'QVQLQQPGAEL'
            ]
        Suppose the regions to be masked are respectively:
            [
                'LVES',
                'QPG'
            ]
        For training, the collated batch will contain (the input_ids
        corresponding to) the prompts:
            [
                '[HEAVY] [HUMAN] E V Q [MASK] I Q P [SEP] [PAD] [PAD]',
                '[HEAVY] [HUMAN] Q V Q L Q [MASK] A E L [SEP]'
            ]
        and the prompts with completions:
            [
                '[HEAVY] [HUMAN] E V Q [MASK] I Q P [SEP] L V E S [CLS] [PAD]',
                '[HEAVY] [HUMAN] Q V Q L Q [MASK] A E L [SEP] Q P G [CLS]'
            ]
        For evaluation, the collated batch only contains the prompt.

        Parameters
        ----------
        batch : dict
            A batch of raw sequence data containing:
                prompt : list of str
                    sequence (heavy chain) to be masked
                LC : list of str
                    the paired light chain
                region : list of str
                    sequence (part of prompt) to be replaced by [MASK] token

        Returns
        -------
        collated_batch : dict
            A batch of collated data containing:
                input_ids : torch.Tensor
                    Input ids to language models of prompts (with completions if eval=False)
                prompts : torch.Tensor
                    Input ids to language models of prompts
                LC : list of str
                    the paired light chain (return only if eval=True)
                seq_pre_mask : list of str
                    sequence before [MASK] token
                seq_post_mask : list of str
                    sequence after [MASK] token
                label : torch.Tensor
                    Mask for completion
                reward : list of float
                    Rewards of the sequences in the batch (return only if eval=False)
        """

        (
            masked_prompts,
            masked_prompts_with_completions,
            spaced_completions,
            sequences_pre_mask,
            sequences_post_mask,
        ) = (
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

            if self.eval:
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

        if self.eval:
            return {
                "input_ids": tokenized_masked_prompts_with_completions,
                "prompts": tokenized_masked_prompts,
                "labels": input_mask,
                "LC": batch["LC"],
                "seq_pre_mask": sequences_pre_mask,
                "seq_post_mask": sequences_post_mask,
            }

        return {
            "input_ids": tokenized_masked_prompts_with_completions,
            "prompts": tokenized_masked_prompts,
            "labels": input_mask,
            "rewards": batch["rewards"],
            "seq_pre_mask": sequences_pre_mask,
            "seq_post_mask": sequences_post_mask,
        }
