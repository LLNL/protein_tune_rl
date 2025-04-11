import torch
import transformers


class AATokenizer:
    def __init__(self, hf_config):

        # print("HF Config", hf_config)
        # os.path.join(trained_models_dir, 'vocab.txt')
        # self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(hf_config + "/")
        self.tokenizer = transformers.BertTokenizerFast(
            vocab_file=hf_config + "/vocab.txt", do_lower_case=False
        )
        self.tokenizer.add_special_tokens(AATokenizer.conditional_tokens())
        self.vocab_size = len(self.tokenizer)

        # print(f"Number of tokens; {self.vocab_size}")
        # print(self.tokenizer)

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
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)

        return input_ids
