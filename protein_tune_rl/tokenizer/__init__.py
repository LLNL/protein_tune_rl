def create_tokenizer(name, tokenizer_config, **kwargs):

    if name=="aa_tokenizer":
        from protein_tune_rl.tokenizer.aa_tokenizer import AATokenizer
        return AATokenizer(tokenizer_config)
    
    if name == "iglm_tokenizer":
        raise NotImplementedError
    
    if name == "multi_mask_tokenizer":
        from protein_tune_rl.tokenizer.multi_mask_tokenizer import MultiMaskTokenizer
        return MultiMaskTokenizer(tokenizer_config)