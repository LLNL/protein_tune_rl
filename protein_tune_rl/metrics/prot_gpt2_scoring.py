import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProtGPT2Scoring:
    def __init__(self):
        self.model_path = "/usr/workspace/vaccines/abag_seq/weights/pretrained/protgpt2"

        self.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model, self.tokenizer = self.load_model(self.model_path, self.device)

    def insert_value_every_n_indices(self, string, n):
        """Inserts a given value into a PyTorch batch tensor every n indices."""
        result = "<|endoftext|>" + "\\n"
        for i, c in enumerate(string):
            result += c
            if (i + 1) % n == 0 and i != len(string) - 1:
                result += "\\n"

        result += "\\n"
        result += "<|endoftext|>"

        return [result]

    def load_model(self, model_path, device):

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        try:
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.resize_token_embeddings(len(tokenizer))

        except Exception as e:
            raise ValueError("Error : Cannot load model") from e

        model = model.to(device)
        model = model.eval()

        return model, tokenizer

    def __call__(
        self,
        sequence,
    ):
        """
        Parameters
        ----------

        Returns
        ----------
        scores

        """
        sequence = sequence["H"]
        batch = self.insert_value_every_n_indices(sequence, n=60)
        # input_ids -> shape (batch_size, sequence_length)
        input_ids = self.tokenizer(batch, padding=True)["input_ids"]
        input_ids = torch.tensor(input_ids, device=self.device)
        # create labels from input ids, and drop the first token
        # the first token will be the start of sequence token, we should not to compute the logp for this token
        # labels -> shape (batch_size, sequence_length-1)
        labels = input_ids[:, 1:].clone()
        # given the batch  will be padded, we need to ignore the pad tokens
        # create a mask to remove the PAD token from the computation
        # logp_mask -> shape (batch_size, sequence_length-1)
        logp_mask = labels != self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
        # run a forward pass on the model and remove the final token
        # the logp for the final token is irrelavant for autoregressive models
        # logits -> shape (batch_size, sequence_length-1, vocab_size)
        logits = self.forward(self.model, input_ids)[:, :-1, :]
        # compute the log softmax for each token
        # logps -> shape (batch_size, sequence_length-1, vocab_size)
        logps = torch.log_softmax(logits, dim=-1)
        # gather the relevant logps the correct label at each position
        # label_logps -> shape (batch_size, sequence_length-1)
        label_logps = torch.gather(logps, dim=-1, index=labels.unsqueeze(2)).squeeze(2)
        # apply the mask to ignore the PAD tokens in the computation
        # label_logps_w_mask -> shape (batch_size, sequence_length-1)
        label_logps_w_mask = label_logps * logp_mask
        # take the mean over the logps to score each sequence over the batch
        # BPE tokenizer requires the mean to normalize over different sequence lengths
        # create a dataframe from dictionary with mutation and scores
        value = label_logps_w_mask.sum(-1)  # / logp_mask.sum(-1)
        return value.item()

    def forward(self, model, input_ids):

        with torch.no_grad():
            logits = model(input_ids=input_ids)["logits"]

        return logits
