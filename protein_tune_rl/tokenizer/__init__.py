def create_tokenizer(name, tokenizer_config, **kwargs):

    if name == "iglm_tokenizer":
        from protein_tune_rl.tokenizer.iglm_tokenizer import IgLMTokenizer

        return IgLMTokenizer(tokenizer_config)

    if name == "multi_mask_tokenizer":
        from protein_tune_rl.tokenizer.multi_mask_tokenizer import MultiMaskTokenizer

        return MultiMaskTokenizer(tokenizer_config)
