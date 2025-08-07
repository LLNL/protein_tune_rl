def create_metric(name):
    """
    Create a metric object based on the given name.
    Args:
        name (str): The name of the metric to create.
    Returns:
        object: An instance of the specified metric class.
    Raises:
        ValueError: If the metric name is not recognized.
    """

    try:
        if name == "sasa":
            from protein_tune_rl.metrics.sasa import SASA

            return SASA

        if name == "folding_confidence":
            from protein_tune_rl.metrics.folding_confidence import FoldingConfidence

            return FoldingConfidence

        if name == "prot_gpt2_scoring":
            from protein_tune_rl.metrics.prot_gpt2_scoring import ProtGPT2Scoring

            return ProtGPT2Scoring

        if name == "progen2_scoring":
            from protein_tune_rl.metrics.progen2_scoring import ProGen2Scoring

            return ProGen2Scoring

        if name == "iglm_scoring":
            from protein_tune_rl.metrics.iglm_scoring import IgLMScoring

            return IgLMScoring

        if name == "ss_perc_sheet":
            from protein_tune_rl.metrics.ss_perc_sheet import PercBetaSheet

            return PercBetaSheet

        if name == "iglm_kl_scoring":
            from protein_tune_rl.metrics.iglm_kl_scoring import IgLMKLScoring

            return IgLMKLScoring

        raise ValueError(f"Unknown metric name: {name}")
    except Exception as e:
        raise RuntimeError(f"Failed to create metric '{name}': {e}") from e
