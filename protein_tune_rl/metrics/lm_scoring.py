from typing import Dict

import torch


class LanguageModelScoring:
    def __init__(self, model, pad_token='<|pad|>'):
        self.tokenizer, self.pad_id = self.init_tokenizer(model, pad_token)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.device = torch.device("cpu")

        model = self.init_model(model)
        model = model.to(self.device)
        self.model = model.eval()

    def init_tokenizer(self, model, pad_token):
        raise NotImplementedError

    def init_model(self, model):
        raise NotImplementedError

    def preprocessing(self, sequences):
        raise NotImplementedError

    def __call__(self, chains: Dict):
        """
        Parameters
        ----------

        Returns
        ----------
        scores

        """

        sequences = [chains["H"]]
        input_ids = self.preprocessing(sequences)
        # input_ids -> shape (batch_size, sequence_length)
        # input_ids = self.tokenizer(batch, padding=True)["input_ids"]
        input_ids = torch.tensor(input_ids, device=self.device)
        # create labels from input ids, and drop the first token
        # the first token will be the start of sequence token, we should not to compute the logp for this token
        # labels -> shape (batch_size, sequence_length-1)
        labels = input_ids[:, 1:].clone()
        # given the batch  will be padded, we need to ignore the pad tokens
        # create a mask to remove the PAD token from the computation
        # logp_mask -> shape (batch_size, sequence_length-1)
        logp_mask = labels != self.pad_id
        # run a forward pass on the model and remove the final token
        # the logp for the final token is irrelavant for autoregressive models
        # logits -> shape (batch_size, sequence_length-1, vocab_size)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids)["logits"][:, :-1, :]
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
        return label_logps_w_mask.sum(-1).cpu().numpy()[0]
