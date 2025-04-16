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

    if name == "dro_value":
        from protein_tune_rl.protein_evaluator.dro_value_evaluator import (
            DROValueEvaluator,
        )

        return DROValueEvaluator

    if name == "online_rl_evaluator":
        from protein_tune_rl.protein_evaluator.online_rl_evaluator import (
            OnlineRLEvaluator,
        )

        return OnlineRLEvaluator
