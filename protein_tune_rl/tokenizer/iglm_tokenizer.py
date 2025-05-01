import torch
import transformers


class IgLMTokenizer:
    def __init__(self, hf_config, padding_side="right"):

        self.padding_side = padding_side
        self.tokenizer = transformers.BertTokenizerFast(
            vocab_file=f"{hf_config}/vocab.txt",
            do_lower_case=False,
            padding_side=padding_side,
        )
        self.tokenizer.add_special_tokens(IgLMTokenizer.conditional_tokens())
        self.vocab_size = len(self.tokenizer)

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @staticmethod
    def conditional_tokens():
        return {
            'additional_special_tokens': [
                '[CAMEL]',
                '[HUMAN]',
                '[MOUSE]',
                '[RABBIT]',
                '[RAT]',
                '[RHESUS]',
                '[HEAVY]',
                '[LIGHT]',
            ]
        }

    @property
    def stop_token_id(self):
        return 2

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def __call__(self, sequence):
        encoding = self.tokenizer(sequence)
        # remove start [CLS] and end [SEP] added by the BERT tokenizer
        # IgLM is not trained with these tokens
        return torch.tensor(encoding["input_ids"][1:-1], dtype=torch.long)
