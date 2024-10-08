

from protein_tune_rl.tokenizer.aa_tokenizer import AATokenizer

class MultiMaskTokenizer(AATokenizer):
    
    def __init__(self, hf_config):
        super().__init__(hf_config)
    