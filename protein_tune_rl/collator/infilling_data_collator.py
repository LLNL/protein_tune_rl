import re
import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


class InfillingCollator(DataCollatorWithPadding):
    """Collator for online RL training or evaluation of IgLM."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
    ):
        assert tokenizer.padding_side == "left"
        self.tokenizer = tokenizer
        self.conditional_tokens = "[HEAVY] [HUMAN] "
        self.mask_token = "[MASK]"

    def __call__(self, batch):
        """Collate raw sequence data for online RL training or evaluation of IgLM.
        Following the IgLM paper, sequences in the batch are masked for infilling.
        The batch of collated sequences is padded on the left.

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
        The collated batch will be (the input_ids corresponding to):
            [
                '[PAD] [PAD] [HEAVY] [HUMAN] E V Q [MASK] I Q P [SEP]',
                '[HEAVY] [HUMAN] Q V Q L Q [MASK] A E L [SEP]'
            ]

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
                    Input ids to language models
                attention_mask : torch.Tensor
                    Attention mask to language models
                position_ids : torch.Tensor
                    Position ids to language models
                init_size : int
                    Initial size of input_ids. input_ids will be augmented during
                    sampling, this helps to identify the augmented portion.
                LC : list of str
                    the paired light chain
                seq_pre_mask : list of str
                    sequence before [MASK] token
                seq_post_mask : list of str
                    sequence after [MASK] token
                masked_seq : list of str
                    the part of the orignal sequence replaced by [MASK] token
        """
        infilling_inputs = []

        output = {"LC": [], "masked_seq": [], "seq_pre_mask": [], "seq_post_mask": []}

        for (
            seq_HC,
            seq_LC,
            masked_seq,
        ) in zip(batch["prompts"], batch["LC"], batch["region"]):

            masked_region_idx = re.search(masked_seq, seq_HC)
            seq_pre_mask = seq_HC[: masked_region_idx.start()]
            seq_post_mask = seq_HC[masked_region_idx.end() :]

            infilling_input = (
                self.conditional_tokens
                + " ".join(seq_pre_mask)
                + " "
                + self.mask_token
                + " "
                + " ".join(seq_post_mask)
            )

            infilling_inputs.append(infilling_input)
            output["seq_pre_mask"].append(seq_pre_mask)
            output["seq_post_mask"].append(seq_post_mask)
            output["masked_seq"].append(masked_seq)
            output["LC"].append(seq_LC)

        # TODO this needs to be resolved. Are we using tokenizer.tokenizer or use the tokenizer we pass.
        tokenized_input = self.tokenizer.tokenizer(infilling_inputs, padding=True)
        input_ids = tokenized_input["input_ids"]
        attention_mask = tokenized_input["attention_mask"]

        for i in range(len(batch["prompts"])):
            num_pads = len(input_ids[i]) - sum(attention_mask[i])
            input_ids[i].pop(num_pads)
            attention_mask[i].pop(num_pads)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        output["input_ids"] = input_ids
        output["attention_mask"] = attention_mask
        output["position_ids"] = position_ids
        output["init_size"] = input_ids.size()[-1]

        return output
