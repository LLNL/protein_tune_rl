import re
from typing import Any, Dict, List

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


class InfillingDataCollator(DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        species_token: str = "[HUMAN]",
        mask_region: str = "HCDR3",
    ):
        assert tokenizer.padding_side == "left"
        self.tokenizer = tokenizer
        self.conditional_tokens = f"{species_token} [HEAVY] "
        self.mask_token = "[MASK]"
        self.mask_region = mask_region

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        infilling_inputs = []

        output = {"LC": [], "masked_seq": [], "seq_pre_mask": [], "seq_post_mask": []}

        for seq in batch:
            seq_HC = seq["HC"]
            masked_seq = seq[self.mask_region]

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
            output["LC"].append(seq["LC"])

        tokenized_input = self.tokenizer(infilling_inputs, padding=True)
        input_ids = tokenized_input["input_ids"]
        attention_mask = tokenized_input["attention_mask"]

        for i, seq in enumerate(batch):
            num_pads = len(input_ids[i]) - sum(attention_mask[i])
            input_ids[i].pop(num_pads)
            attention_mask[i].pop(num_pads)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        model_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        output["model_input"] = model_input
        output["init_size"] = input_ids.size()[-1]

        return output
