def create_collator(name, tokenizer, eval=False):
    if name == "dro_infilling":
        from protein_tune_rl.collator.dro_collator import DROCollator

        return DROCollator(tokenizer=tokenizer, eval=eval)

    if name == "infilling":
        from protein_tune_rl.collator.infilling_data_collator import \
            InfillingCollator

        return InfillingCollator(tokenizer=tokenizer)
