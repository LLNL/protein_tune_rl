import logging
from pathlib import Path

from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
import transformers
import torch
from amino_acid_tokenizer import AminoAcidTokenizer

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    def __init__(self, model, name):
        super(Decoder, self).__init__()
        self.model = model
        self.name = name


    def forward(self, input_ids, token_type_ids=None, labels=None, **kwargs) -> CausalLMOutput:

        output = self.model(input_ids=input_ids,
                            labels=labels, return_dict=True)
        return output
    
    


