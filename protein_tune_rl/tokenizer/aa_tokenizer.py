import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
import transformers


class AminoAcidTokenizer:
    def __init__(self, hf_config):

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(hf_config)
        self.vocab_size = len(self.tokenizer)

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def pad_token_id(self):            
        return self.tokenizer.pad_token_id
    

    def __call__(self, sequence):
        encoding = self.tokenizer(sequence, **self.kwargs)
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)           
        
        return input_ids
    

    
