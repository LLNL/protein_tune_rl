# protein_tune_rl/metrics/iglm_tokenwise_kl.py
from __future__ import annotations

from typing import Dict, Tuple, Union, List

import torch
import torch.nn.functional as F

from protein_tune_rl.metrics.iglm_scoring import IgLMScoring, get_seq_length
from protein_tune_rl import logger


Tensor = torch.Tensor


def _safe_softmax(logits: Tensor, dim: int) -> Tensor:
    # numerically stable softmax
    return F.softmax(logits.float(), dim=dim)


def _kl_categorical_pq(p: Tensor, q: Tensor, dim: int = -1) -> Tensor:
    """
    KL(p || q) for categorical distributions with probability vectors p and q.
    Assumes p and q are proper probability distributions along `dim`.
    Returns KL for each slice along `dim` (i.e., reduces over `dim`).
    """
    eps = torch.finfo(p.dtype).eps if p.is_floating_point() else 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (p.log() - q.log()), dim=dim)


class IgLMTokenwiseKLDivergence:
    """
    Token-wise KL divergence metric for IgLM policies.

    This metric computes, for each *infill* position l in the sequence, the
    KL divergence between the primary policy's next-token distribution and the
    reference policy's next-token distribution:

        KL_t(l) = sum_v  p_theta(v | x, l) * [ log p_theta(v | x, l) - log p_ref(v | x, l) ]

    It returns:
      - a vector of per-step KL values over the infill region, and
      - the average KL normalized by the number of infilled positions.

    If logits (or full distributions) are not available from IgLMScoring,
    the metric falls back to a Monte Carlo estimate using the realized tokens y:

        KL(p_theta || p_ref) ≈ sum_l [ log p_theta(y_l | x) - log p_ref(y_l | x) ] / L

    Parameters
    ----------
    model : Any
        The primary policy/model (pi_theta).
    ref_model : Any
        The reference policy/model (pi_ref).
    tokenizer : Any
        Tokenizer compatible with IgLMScoring.
    reduction : {"none", "mean"}
        How to reduce the per-step KL when returning a scalar summary. "mean"
        returns length-normalized mean; "none" returns only the vector.
    return_both : bool
        If True, return both the per-step vector and the scalar summary.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        reduction: str = "mean",
        return_both: bool = True,
    ) -> None:
        self.reduction = reduction
        self.return_both = return_both

        # Wrap models with existing scoring helper
        self.primary = IgLMScoring(model, tokenizer)
        self.reference = IgLMScoring(ref_model, tokenizer)

        # Probe for logits API once; cache capability
        self._has_logits_api = any(
            hasattr(sc, attr)
            for sc in (self.primary, self.reference)
            for attr in (
                "logits",
                "get_logits",
                "token_logits",
                "distribution_over_infill",
            )
        )
        if not self._has_logits_api:
            logger.warning(
                "[IgLMTokenwiseKLDivergence] IgLMScoring does not expose logits. "
                "Falling back to Monte-Carlo (realized token) KL estimate."
            )

    def update_model(self, new_model) -> None:
        """Update the primary model (e.g., during online RL)."""
        logger.info("Updating primary IgLM model in Tokenwise KL metric")
        self.primary.update_model(new_model)

    def _infer_infill_range(self, chains: Dict) -> Tuple[int, int]:

        chains_seq_pre_mask_length = get_seq_length(chains["seq_pre_mask"])
        chains_seq_post_mask_length = get_seq_length(chains["seq_post_mask"])
        chains_H_length = get_seq_length(chains["H"])

        # Same convention as your existing scoring utilities
        return (
            chains_seq_pre_mask_length,
            chains_H_length - chains_seq_post_mask_length,
        )

    def _fetch_logits_over_infill(
        self,
        scoring: IgLMScoring,
        seq: Union[str, List[str]],
        chain_token: str,
        species_token: str,
        infill_range: Tuple[int, int],
    ) -> Tensor:
        """
        Try several commonly used method names to get next-token logits over the infill region.
        Expected shape: [L, V] where L is infill length and V is vocab size.
        """
        # Try a few attr names to keep this robust to minor refactors
        for attr in (
            "token_logits_over_infill",  # preferred if available
            "logits",
            "get_logits",
            "token_logits",
            "distribution_over_infill",
        ):
            if hasattr(scoring, attr):
                fn = getattr(scoring, attr)
                logits = fn(
                    seq,
                    chain_token,
                    species_token,
                    infill_range=infill_range,
                )
                # Some implementations might directly return probabilities;
                # if they sum to ~1 along last dim, treat as probs and convert to logits.
                if logits.dim() >= 2:
                    s = logits.sum(dim=-1, keepdim=True)
                    if torch.allclose(s, torch.ones_like(s), atol=1e-4, rtol=1e-4):
                        # convert probs -> logits safely
                        eps = (
                            torch.finfo(logits.dtype).eps
                            if logits.is_floating_point()
                            else 1e-12
                        )
                        logits = (logits.clamp_min(eps)).log()
                return logits

        raise AttributeError("No logits-bearing method found on IgLMScoring.")

    def __call__(self, chains: Dict):
        """
        Compute token-wise KL over the infill region and (optionally) the length-normalized mean.

        Returns
        -------
        If return_both:
            dict with keys:
                "per_step_kl": Tensor [L]  (KL at each infill position)
                "mean_kl": float           (KL averaged over L)
        Else:
            If reduction == "mean": float
            If reduction == "none": Tensor [L]
        """
        infill_range = self._infer_infill_range(chains)
        chain_token = "[HEAVY]"
        species_token = "[HUMAN]"

        if self._has_logits_api:
            # Full distribution KL per step
            primary_logits = self._fetch_logits_over_infill(
                self.primary, chains["H"], chain_token, species_token, infill_range
            )
            reference_logits = self._fetch_logits_over_infill(
                self.reference, chains["H"], chain_token, species_token, infill_range
            )

            if primary_logits.shape != reference_logits.shape:
                raise ValueError(
                    f"Primary and reference logits have mismatched shapes: "
                    f"{tuple(primary_logits.shape)} vs {tuple(reference_logits.shape)}"
                )

            p = _safe_softmax(primary_logits, dim=-1)  # [L, V]
            q = _safe_softmax(reference_logits, dim=-1)  # [L, V]
            per_step_kl = _kl_categorical_pq(p, q, dim=-1)  # [L]

        else:
            # Fallback: Monte-Carlo KL estimate using realized tokens
            # This is equivalent to the log-likelihood difference approach.
            # NOTE: This *does not* capture full distribution KL per step.
            logger.warning(
                "[IgLMTokenwiseKLDivergence] Using Monte-Carlo KL estimate "
                "(log-likelihood difference) as fallback."
            )
            primary_ll = self.primary.log_likelihood(
                chains["H"],
                chain_token,
                species_token,
                infill_range=infill_range,
                reduction="none",  # get tokenwise
            )  # shape [L]
            ref_ll = self.reference.log_likelihood(
                chains["H"],
                chain_token,
                species_token,
                infill_range=infill_range,
                reduction="none",
            )  # shape [L]
            per_step_kl = (primary_ll - ref_ll).detach()

        # Reduction / return format
        L = per_step_kl.shape[0]
        mean_kl = per_step_kl.mean().item() if L > 0 else 0.0

        if self.return_both:
            return {"per_step_kl": per_step_kl, "mean_kl": mean_kl}

        if self.reduction == "mean":
            return mean_kl
        elif self.reduction == "none":
            return per_step_kl
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction!r}")
