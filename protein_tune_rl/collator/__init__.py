def create_collator(name, model_name=None, tokenizer=None, **kwargs):
    if name == "dro_infilling":
        from protein_tune_rl.collator.dro_collator import DRODataCollator

        return DRODataCollator(model_name=model_name, tokenizer=tokenizer)

    if name == "infilling":
        from optlm.protein_tune_rl.protein_tune_rl.collator.iglm_data_collator import \
            InfillingDataCollator

        return InfillingDataCollator(tokenizer=tokenizer, **kwargs)
