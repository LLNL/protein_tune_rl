import torch
from tokenizers import Encoding, Tokenizer
from transformers import AutoModelForCausalLM

from protein_tune_rl.metrics.lm_scoring import LanguageModelScoring


class ProGen2Scoring(LanguageModelScoring):
    def __init__(self, model="hugohrban/progen2-small"):
        super().__init__(model, pad_token='<|pad|>')

    def init_tokenizer(self, model, pad_token):
        tokenizer = Tokenizer.from_pretrained(model)
        tokenizer.enable_padding(pad_token=pad_token)
        pad_id = tokenizer.token_to_id(pad_token)
        return tokenizer, pad_id

    def init_model(self, model):
        return AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)

    def preprocessing(self, sequences):
        encoding: Encoding = self.tokenizer.encode_batch(sequences)
        return torch.tensor([e.ids for e in encoding])
