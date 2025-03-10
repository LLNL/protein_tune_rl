def create_collator(name, model_name, tokenizer):
    if name == "dro":
        from protein_tune_rl.collator.dro_collator import DROCollator

        return DROCollator(model_name=model_name, tokenizer=tokenizer)
