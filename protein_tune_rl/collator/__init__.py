def create_collator(name, model_name=None, tokenizer=None, **kwargs):
    if name == "dro":
        from protein_tune_rl.collator.dro_collator import DROCollator

        return DROCollator(model_name=model_name, tokenizer=tokenizer)

    if name == "infilling":
        from protein_tune_rl.collator.infilling_data_collator import \
            InfillingDataCollator

        return InfillingDataCollator(tokenizer=tokenizer, **kwargs)
