def create_metric(name):
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

    if name == "ss_perc_sheet":
        from protein_tune_rl.metrics.ss_perc_sheet import PercBetaSheet

        return PercBetaSheet
