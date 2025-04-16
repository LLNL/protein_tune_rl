import logging
import math

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutput

logger = logging.getLogger(__name__)


class Decoder(nn.Module):
    def __init__(self, model, name):
        super(Decoder, self).__init__()
        self.model = model
        self.name = name

    def forward(
        self, input_ids, labels=None, attention_mask=None, **kwargs
    ) -> CausalLMOutput:

        return self.model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            attention_mask=attention_mask,
            **kwargs,
        )

    def save(self, path) -> None:
        self.model.save_pretrained(path)


class DecoderWithLinearHead(nn.Module):
    def __init__(self, model, name, train_all_params):
        super().__init__()
        self.model = model
        self.train_all_params = train_all_params
        self.name = name

        # If not training all parameters, freeze the parameters of the base LLM.
        # As a result, only the parameters of the linear head are trainable
        if not self.train_all_params:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # Compute the input size of the linear head and the number of layers in the stack.
        in_features = self.model.lm_head.in_features
        linear_relu_stack = []
        num_layers = math.ceil(-5 + math.log(in_features) / math.log(2))
        for i in range(num_layers):
            linear_relu_stack.extend(
                (
                    nn.Linear(in_features // 2**i, in_features // 2 ** (i + 1)),
                    nn.ReLU(),
                )
            )
        linear_relu_stack.append(nn.Linear(in_features // 2**num_layers, 1))
        self.linear_relu_stack = nn.Sequential(*linear_relu_stack)

    def forward(
        self, input_ids, labels=None, attention_mask=None, **kwargs
    ) -> CausalLMOutput:

        if not self.train_all_params:
            with torch.no_grad():
                output = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True,
                    attention_mask=attention_mask,
                )
        else:
            output = self.model(
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                attention_mask=attention_mask,
            )

        # Tensor shape (batch_size, sequence_length, hidden_dimension_size)
        last_hidden_state = output['hidden_states'][-1]
        # Tensor shape (batch_size, hidden_dimension_size)
        decoder_cls = last_hidden_state[:, :, :].mean(1)

        return self.linear_relu_stack(decoder_cls).float()
