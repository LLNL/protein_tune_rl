from optlm.protein_tune_rl.protein_tune_rl.tokenizer.iglm_tokenizer import \
    AATokenizer


class MultiMaskTokenizer(AATokenizer):
    def __init__(self, hf_config):
        super().__init__(hf_config)
