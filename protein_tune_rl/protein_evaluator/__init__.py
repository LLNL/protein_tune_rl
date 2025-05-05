def create_evaluator(name):
    if name == "test":
        from protein_tune_rl.protein_trainer.test_trainer import TestTrainer

        return TestTrainer

    if name == "sasa":
        from protein_tune_rl.protein_evaluator.iglm_evaluator import \
            IGLMEvaluator

        return IGLMEvaluator

    if name == "ss_perc_sheet":
        from protein_tune_rl.protein_evaluator.iglm_evaluator import \
            IGLMEvaluator

        return IGLMEvaluator

    if name == "dro_value":
        from protein_tune_rl.protein_evaluator.dro_value_evaluator import \
            DROValueEvaluator

        return DROValueEvaluator
    
    if "sequence" in name:
        from protein_tune_rl.protein_evaluator.sequence_evaluator import \
            SequenceEvaluator
        return SequenceEvaluator 

