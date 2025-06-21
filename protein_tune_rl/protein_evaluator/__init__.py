def create_evaluator(name):
    """
    There are three types:
        - iglm: for evaluating fine-tuned or original IgLMs
        - sequence: for evaluating sequences from a dataset
        - dro_value: for evaluating sequences from a dataset using DRO value network as metric

    For 'iglm' and 'sequence', the evaluation can be done on multiple metrics
    like ss_perc_sheet, SASA, and LM scorings (specified in the config), while
    the metric for 'dro_value' is fixed (the DRO value network).
    """

    try:
        if name == "iglm":
            from protein_tune_rl.protein_evaluator.iglm_evaluator import IGLMEvaluator

            return IGLMEvaluator

        if name == "dro_value":
            from protein_tune_rl.protein_evaluator.dro_value_evaluator import (
                DROValueEvaluator,
            )

            return DROValueEvaluator

        if name == "sequence":
            from protein_tune_rl.protein_evaluator.sequence_evaluator import (
                SequenceEvaluator,
            )

            return SequenceEvaluator

        raise ValueError(f"Unknown evaluator name: {name}")
    except Exception as e:
        raise RuntimeError(f"Failed to create evaluator '{name}': {e}") from e
