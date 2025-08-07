from typing import Dict
from protein_tune_rl.metrics.iglm_scoring import IgLMScoring
from protein_tune_rl import logger


class IgLMKLScoring:
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        return_only_ref_model_scores=False,
        reduction="sum",
    ):
        """
        Parameters
        ----------
        path : str
            Path or identifier for the primary model (pi_theta).
        ref_path : str
            Path or identifier for the reference model (pi_ref).
        """

        self.return_only_ref_model_scores = return_only_ref_model_scores
        self.reduction = reduction

        if not self.return_only_ref_model_scores:
            # Initialize the primary model
            self.primary_IgLMScoring = IgLMScoring(model, tokenizer)

        # Initialize the reference model
        self.reference_IgLMScoring = IgLMScoring(ref_model, tokenizer)

    def update_model(self, new_model):
        """
        Replace the current scoring model with a new one (e.g., the current training policy).
        """
        if not self.return_only_ref_model_scores:
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
            KL(p_theta(x) || p_ref(x)) \approx \frac{1}{N} sum_(i=1)^N score(y_i) with y_i ~ p_theta(x)

        Recall that the KL divergence is defined as:
        .. math::
            KL(p_theta(x) || p_ref(x)) = E_{p_theta(x)}[log p_theta(x) - log p_ref(x)]
        """

        infill_range = (
            len(chains["seq_pre_mask"]),
            len(chains["H"]) - len(chains["seq_post_mask"]),
        )

        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"

        reference_scores = self.reference_IgLMScoring.log_likelihood(
            chains["H"],
            chain_token,
            species_token,
            infill_range=infill_range,
            reduction=self.reduction,
        )

        if self.return_only_ref_model_scores:
            return reference_scores

        return (
            self.primary_IgLMScoring.log_likelihood(
                chains["H"],
                chain_token,
                species_token,
                infill_range=infill_range,
                reduction=self.reduction,
            )
            - reference_scores
        )
