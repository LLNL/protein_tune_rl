def create_evaluator(name):
    """Create and return an evaluator class based on the specified name.

    This factory function instantiates different types of protein sequence evaluators
    for various evaluation scenarios in protein design and optimization.

    Args:
        name (str): The type of evaluator to create. Must be one of:
            - "iglm": For evaluating fine-tuned or original IgLM models
            - "sequence": For evaluating sequences from a dataset
            - "dro_value": For evaluating sequences using DRO value network as metric

    Returns:
        class: The evaluator class (not an instance) that can be instantiated with
               appropriate configuration parameters.

    Raises:
        ValueError: If the evaluator name is not recognized.
        RuntimeError: If there's an error importing or creating the evaluator.

    Note:
        - For 'iglm' and 'sequence' evaluators, evaluation can be performed using
          multiple metrics (e.g., ss_perc_sheet, SASA, LM scorings) as specified
          in the configuration.
        - For 'dro_value' evaluator, the metric is fixed to the DRO value network.

    Example:
        >>> evaluator_class = create_evaluator("iglm")
        >>> evaluator = evaluator_class(config)
        >>> results = evaluator.run(output_dir)
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
