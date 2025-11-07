import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from protein_tune_rl import logger
from protein_tune_rl.collator import create_collator
from protein_tune_rl.dataloader import create_dataloader
from protein_tune_rl.dataset import create_dataset
from protein_tune_rl.models import create_model
from protein_tune_rl.optimizer.dpo import DPO  # Import the DPO optimizer
from protein_tune_rl.protein_trainer import create_optimizer
from protein_tune_rl.protein_trainer.trainer import Trainer
from protein_tune_rl.tokenizer import create_tokenizer


class DPOTrainer(Trainer):
    def __init__(self, config):
        """Initialize the DPO Trainer with the provided configuration."""
        super().__init__(config)

        # Training hyperparameters
        self.total_optimization_steps = config["trainer"]["total_optimization_steps"]
        self.batch_size = config["trainer"]["batch_size"]
        self.learning_rate = config["trainer"]["learning_rate"]
        self.check_point_freq = config["trainer"]["check_point_freq"]
        self.beta = config["trainer"].get("dpo_beta", 0.1)  # DPO β parameter

        # Load dataset of preference pairs
        self.dataset = create_dataset(
            name=config["dataset"]["name"],
            # Use provided pair dataset path (could be JSONL/CSV/Parquet)
            data_directory=config["dataset"].get(
                "pair_dataset", config["dataset"].get("data_directory")
            ),
        )

        # Initialize tokenizer and collator
        self.tokenizer = create_tokenizer(
            name=config["tokenizer"]["name"],
            tokenizer_config=config["tokenizer"]["tokenizer_config"],
        )
        self.collator = create_collator(
            name=config["collator"]["name"],
            tokenizer=self.tokenizer,
        )

        # DataLoader (shuffling could be True for training; keeping False to iterate deterministically)
        self.dataloader = create_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

        # Create policy model (trainable) and reference model (frozen)
        self.policy = create_model(
            name="iglm",
            hf_config=config["policy_model"]["dir"],
            vocab_size=self.tokenizer.vocab_size,
        ).to(self.device)
        self._maybe_load_state_dict(self.policy, "policy")
        self.policy = DDP(self.policy, device_ids=self.device_ids)

        # Reference model: load from specified checkpoint or same as policy initialization
        ref_model_path = config.get("reference_model", {}).get(
            "dir", config["policy_model"]["dir"]
        )
        self.reference = create_model(
            name="iglm",
            hf_config=ref_model_path,
            vocab_size=self.tokenizer.vocab_size,
        ).to(self.device)
        self.reference = DDP(self.reference, device_ids=self.device_ids)
        self.reference.eval()  # freeze reference model

        # Initialize DPO optimizer module
        self.model_optimizer = DPO(
            policy=self.policy,
            reference=self.reference,
            tokenizer=self.tokenizer,
            device=self.device,
            beta=self.beta,
            length_normalize=config["trainer"].get("length_normalize", False),
        )

        # Setup optimizer for policy model parameters
        self.optimizer_class = create_optimizer(config["trainer"]["optimizer"])
        self.policy_optimizer = self.optimizer_class(
            self.policy.parameters(), lr=self.learning_rate
        )
        self._maybe_load_state_dict(self.policy_optimizer, "policy_optimizer")

        # Enable evaluator if configured (e.g., to monitor generation quality during training)
        if config["trainer"].get("evaluate_during_training", False):
            logger.info(
                "Online evaluation enabled. Evaluator will run during training."
            )
            from protein_tune_rl.protein_evaluator.iglm_evaluator import IGLMEvaluator

            # If the config has a collator_eval section, use it to create a separate collator for evaluation
            if "collator_eval" in config:
                config["collator"] = config["collator_eval"]

            eval_policy = self._unwrap_ddp_model(self.policy)
            self.evaluator = IGLMEvaluator(config, policy_model=eval_policy)

    def save_models(self, output_dir, current_step):
        """Save model checkpoints (policy state dict and full model)."""
        # Save state dict
        torch.save(
            self.policy.state_dict(),
            f"{output_dir}/policy_model_step_{current_step}.bin",
        )
        # Save full model (unwrapped) for easy re-loading with tokenizer
        self.policy.module.save(output_dir / f"models/batch{current_step}")
        logger.info(f"Policy model saved at step {current_step} to {output_dir}.")

    def run_evaluation(self, output_dir, current_step):
        """Run evaluation with the current policy model (if evaluator is set)."""
        logger.info(f"Running evaluation at step {current_step}...")
        eval_policy = self._unwrap_ddp_model(self.policy)
        self.evaluator.update_policy(eval_policy)
        with torch.no_grad():
            eval_df = self.evaluator.run_with_ground_truth()
        if dist.get_rank() == 0 and eval_df is not None:
            eval_df = eval_df.sort_values("__row_idx__").reset_index(drop=True)
            eval_df.to_csv(
                f"{output_dir}/evaluation_results_step_{current_step}.csv", index=False
            )
        dist.barrier()
        logger.info(f"Evaluation complete at step {current_step}.")

    def run(self, output_dir):
        """Execute the training loop for DPO."""
        log_df = pd.DataFrame()
        self._log_dataset_info(self.dataloader, logger)

        current_step = 0
        # Iterate until reaching total optimization steps
        while current_step < self.total_optimization_steps:
            for batch_idx, raw_batch in enumerate(self.dataloader):
                if self.ckpt and current_step < self.ckpt["step"]:
                    current_step += 1
                    continue

                # Prepare batch (tokenize and collate)
                batch = self.collator(raw_batch)
                # Perform one training step
                current_step = self._train_step(batch, current_step, batch_idx)
                # Log the step results
                log_df = self._log_step(log_df, output_dir, current_step)

                # Checkpointing and evaluation
                if self._should_checkpoint(current_step, self.check_point_freq):
                    # Log the step results
                    if dist.get_rank() == 0:
                        log_df.to_csv(f"{output_dir}/dpo_trainer_log.csv", index=False)
                    dist.barrier()
                    # Save model checkpoints
                    if self.config["trainer"].get("save_models", True):
                        self._save_checkpoint(output_dir, "dpo", current_step)
                    if self.config["trainer"].get("evaluate_during_training", False):
                        # Only run evaluation on one rank (rank 0) to avoid duplication
                        self.run_evaluation(output_dir, current_step)

                if current_step >= self.total_optimization_steps:
                    break

        # Final model save
        if current_step % self.check_point_freq:
            self._save_checkpoint(output_dir, "dpo", current_step)
        return log_df

    def _train_step(self, batch, current_step, batch_number):
        """Perform a single optimization step on the given batch."""
        self.policy.train()
        self.policy_optimizer.zero_grad()

        # Calculate DPO loss
        (
            policy_loss,
            diff,
            pi_pos,
            pi_neg,
            ref_pos,
            ref_neg,
        ) = self.model_optimizer.calculate_loss(batch)
        policy_loss.backward()
        self.policy_optimizer.step()

        # Compute training metrics
        # Pairwise accuracy: fraction of pairs where positive log-prob > negative log-prob
        pairwise_acc = (diff > 0).float().mean().item()
        avg_margin = diff.mean().item()

        pairwise_pi_acc = (pi_pos > pi_neg).float().mean().item()
        avg_pi_margin = (pi_pos - pi_neg).mean().item()
        pairwise_ref_acc = (ref_pos > ref_neg).float().mean().item()
        avg_ref_margin = (ref_pos - ref_neg).mean().item()

        logger.info(
            f"Step {current_step + 1}, Batch {batch_number + 1}: "
            f"DPO Loss = {policy_loss.item():.4f}, Pairwise Acc = {pairwise_acc*100:.1f}%, Avg Margin = {avg_margin:.4f}"
        )

        # Store latest values for logging
        self._last_policy_loss = policy_loss.item()
        self._last_pairwise_acc = pairwise_acc
        self._last_avg_margin = avg_margin
        self._last_pairwise_pi_acc = pairwise_pi_acc
        self._last_avg_pi_margin = avg_pi_margin
        self._last_pairwise_ref_acc = pairwise_ref_acc
        self._last_avg_ref_margin = avg_ref_margin
        return current_step + 1

    def _log_step(self, log_df, output_dir, current_step):
        """Record training progress to a CSV log (on rank 0)."""
        if dist.get_rank() == 0:
            step_data = {
                "step": current_step,
                "policy_loss": self._last_policy_loss,
                "pairwise_accuracy": self._last_pairwise_acc,
                "avg_margin": self._last_avg_margin,
                "pairwise_pi_accuracy": self._last_pairwise_pi_acc,
                "avg_pi_margin": self._last_avg_pi_margin,
                "pairwise_ref_accuracy": self._last_pairwise_ref_acc,
                "avg_ref_margin": self._last_avg_ref_margin,
            }
            log_df = pd.concat([log_df, pd.DataFrame([step_data])], ignore_index=True)
        return log_df
