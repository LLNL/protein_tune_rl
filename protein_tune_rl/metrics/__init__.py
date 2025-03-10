def create_metric(name):
    if name == "sasa":
        from protein_tune_rl.metrics.sasa import SASA

        return SASA

    if name == "prot_gpt2_scoring":
        from protein_tune_rl.metrics.prot_gpt2_scoring import ProtGPT2Scoring

        return ProtGPT2Scoring

    if name == "ss_perc_sheet":
        from protein_tune_rl.metrics.ss_perc_sheet import PercBetaSheet

        return PercBetaSheet
