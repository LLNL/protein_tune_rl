from transformers import AutoModelForCausalLM, AutoTokenizer

from protein_tune_rl.metrics.lm_scoring import LanguageModelScoring


class ProtGPT2Scoring(LanguageModelScoring):
    def __init__(
        self, path="/usr/workspace/vaccines/abag_seq/weights/pretrained/protgpt2"
    ):
        super().__init__(path, pad_token='[PAD]')
        self.newline_distance = 60

    def init_tokenizer(self, model, pad_token):
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.add_special_tokens({'pad_token': pad_token})
        pad_id = tokenizer.convert_tokens_to_ids(pad_token)
        return tokenizer, pad_id

    def init_model(self, model):
        try:
            model = AutoModelForCausalLM.from_pretrained(model)
            model.resize_token_embeddings(len(self.tokenizer))
        except ValueError as e:
            raise ValueError(f"Error : Cannot load model {model}") from e

        return model

    def preprocessing(self, sequences):
        """Inserts a given value into a PyTorch batch tensor every n indices."""
        batch = []
        for seq in sequences:
            result = "<|endoftext|>" + "\\n"
            for i, c in enumerate(seq):
                result += c
                if (i + 1) % self.newline_distance == 0 and i != len(seq) - 1:
                    result += "\\n"

            result += "\\n"
            result += "<|endoftext|>"
            batch.append(result)

        return self.tokenizer(batch, padding=True)["input_ids"]
