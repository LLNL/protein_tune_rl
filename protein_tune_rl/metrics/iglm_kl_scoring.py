from typing import Dict
from protein_tune_rl.metrics.iglm_scoring import IgLMScoring
from protein_tune_rl import logger


class IgLMKLScoring:
    def __init__(self, model, ref_model, tokenizer):
        """
        Parameters
        ----------
        path : str
            Path or identifier for the primary model (pi_theta).
        ref_path : str
            Path or identifier for the reference model (pi_ref).
        """

        # Initialize the primary model
        self.primary_IgLMScoring = IgLMScoring(model, tokenizer)

        # Initialize the reference model
        self.reference_IgLMScoring = IgLMScoring(ref_model, tokenizer)

    def update_model(self, new_model):
        """
        Replace the current scoring model with a new one (e.g., the current training policy).
        """
        logger.info("Updating IGLM model in KL scoring function")
        self.primary_IgLMScoring.update_model(new_model)

    def __call__(self, chains: Dict):
        """
        This function computes the folowing score:
        .. math::
            score(y) = sum_{l=1}^L log p_theta(y_l | x) - sum_{l=1}^L log p_ref(y_l | x)

        where :math:`p_theta` is the primary model and :math:`p_ref` is the reference model.
        Note that the score can be then used to compute an approximate KL divergence using:
        .. math::
            KL(p_theta(x) || p_ref(x)) = \frac{1}{N} sum_(i=1)^N score(y_i) with y_i ~ p_theta(x)
        """

        infill_range = (
            len(chains["seq_pre_mask"]),
            len(chains["H"]) - len(chains["seq_post_mask"]),
        )

        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"

        return self.primary_IgLMScoring.log_likelihood(
            chains["H"],
            chain_token,
            species_token,
            infill_range=infill_range,
            reduction="sum",
        ) - self.reference_IgLMScoring.log_likelihood(
            chains["H"],
            chain_token,
            species_token,
            infill_range=infill_range,
            reduction="sum",
        )
