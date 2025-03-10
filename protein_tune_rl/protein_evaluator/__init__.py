def create_evaluator(name):
    if name == "test":
        from protein_tune_rl.protein_trainer.test_trainer import TestTrainer

        return TestTrainer

    if name == "sasa":
        from protein_tune_rl.protein_evaluator.dro_evaluator import DROEvaluator

        return DROEvaluator

    if name == "ss_perc_sheet":
        from protein_tune_rl.protein_evaluator.dro_evaluator import DROEvaluator

        return DROEvaluator
