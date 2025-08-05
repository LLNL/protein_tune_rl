import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from protein_tune_rl import logger
from protein_tune_rl.collator import create_collator
from protein_tune_rl.dataloader import create_dataloader
from protein_tune_rl.dataset import create_dataset
from protein_tune_rl.models import create_model
from protein_tune_rl.optimizer.dro import DRO
from protein_tune_rl.protein_trainer import create_optimizer
from protein_tune_rl.protein_trainer.trainer import Trainer
from protein_tune_rl.tokenizer import create_tokenizer


class DROTrainer(Trainer):
    def __init__(self, config):
        """Initialize the DRO Trainer with the provided configuration.

        Args:
            config (dict): Configuration dictionary containing parameters for training.
        """
        super().__init__(config)

        # Store the configuration.
        self.config = config

        # Catch with device is available.
        if torch.cuda.is_available():
            self.device_ids = [torch.cuda.current_device()]
            self.device = torch.device("cuda", self.device_ids[0])
        else:
            self.device_ids = None
            self.device = torch.device("cpu")

        self.total_optimization_steps = self.config["trainer"][
            "total_optimization_steps"
        ]
        self.batch_size = self.config["trainer"]["batch_size"]
        self.learning_rate = self.config["trainer"]["learning_rate"]
        self.check_point_freq = self.config["trainer"]["check_point_freq"]
        self.train_all_value_params = self.config['value_model']['train_all_params']

        self.dataset = create_dataset(
            name=self.config['dataset']['name'],
            data_directory=self.config['dataset']['data_directory'],
            chain=self.config["dataset"]["chain"],
            region=self.config["dataset"]["region"],
            reward=self.config["dataset"]["reward"],
        )

        self.tokenizer = create_tokenizer(
            name=self.config['tokenizer']['name'],
            tokenizer_config=self.config['tokenizer']['tokenizer_config'],
        )

        self.collator = create_collator(
            name=self.config['collator']['name'],
            tokenizer=self.tokenizer,
        )

        self.dataloader = create_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

        self.policy = create_model(
            name="iglm",
            hf_config=self.config['policy_model']['dir'],
            vocab_size=self.tokenizer.vocab_size,
        ).to(self.device)
        self.policy = DDP(self.policy, device_ids=self.device_ids)

        self.reference = create_model(
            name="iglm",
            hf_config=self.config['policy_model']['dir'],
            vocab_size=self.tokenizer.vocab_size,
        ).to(self.device)
        self.reference = DDP(self.reference, device_ids=self.device_ids)

        self.value = create_model(
            name="iglm_w_linear_head",
            hf_config=self.config['value_model']['dir'],
            vocab_size=self.tokenizer.vocab_size,
            train_all_params=self.train_all_value_params,
        ).to(self.device)
        self.value = DDP(self.value, device_ids=self.device_ids)

        self.reference.eval()

        self.model_optimizer = DRO(
            policy=self.policy,
            reference=self.reference,
            value=self.value,
            tokenizer=self.tokenizer,
            device=self.device,
            tau=self.config["trainer"]["tau"],
            mean=self.config["trainer"]["mean_loss"],
            rescaling=self.config["trainer"]["rescaling"],
            reward_rescaling=self.config["trainer"].get("reward_rescaling", 1.0),
        )

        # Initialize the optimizer
        self.optimizer_class = create_optimizer(self.config["trainer"]["optimizer"])
        self.policy_optimizer = self.optimizer_class(
            self.policy.parameters(), lr=self.learning_rate
        )
        self.value_optimizer = self.optimizer_class(
            self.value.parameters(), lr=self.learning_rate
        )

        if self.config["trainer"].get("evaluate_during_training", False):
            logger.info(
                "Online evaluation is enabled. Evaluator will be run during training."
            )

            # Instantiate the evaluator if online evaluation is enabled
            from protein_tune_rl.protein_evaluator.iglm_evaluator import IGLMEvaluator

            # If DDP-wrapped, pass .module (unwrap the DDP model)
            eval_policy = self._unwrap_ddp_model(self.policy)
            self.evaluator = IGLMEvaluator(self.config, policy_model=eval_policy)

    def _unwrap_ddp_model(self, model):
        """Unwrap DDP model to get the underlying model."""
        return (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )

    def save_models(self, output_dir, current_step):
        """Save both state dict and full model checkpoints."""

        # Only the main (rank 0) process saves the checkpoints
        if dist.get_rank() == 0:

            # Ensure all processes are ready for checkpointing
            dist.barrier()

            # Save DDP state dicts and full (unwrapped) model
            torch.save(
                self.policy.state_dict(),
                f"{output_dir}/policy_model_step_{current_step}.bin",
            )
            torch.save(
                self.value.state_dict(),
                f"{output_dir}/value_model_step_{current_step}.bin",
            )

            # Full Model Save
            self.policy.module.save(output_dir / f"models/batch{current_step}")

            # Signal other ranks to proceed
            dist.barrier()
        else:
            # Non-zero ranks wait for checkpoint to complete
            dist.barrier()
            dist.barrier()

        logger.info(f"Models saved at step {current_step} to {output_dir}.")

    def run_evaluation(self, output_dir, current_step):
        """Run evaluation at the current training step."""
        logger.info(f"Running evaluation at step {current_step}...")

        # If DDP-wrapped, pass .module (unwrap the DDP model)
        eval_policy = self._unwrap_ddp_model(self.policy)

        # Update evaluator with the current policy model
        self.evaluator.update_policy(eval_policy)

        with torch.no_grad():
            eval_df = self.evaluator.run_with_ground_truth(output_dir)

        if eval_df is not None:
            eval_df.to_csv(
                f"{output_dir}/evaluation_results_step_{current_step}.csv",
                index=False,
            )
        else:
            logger.info(f"Rank {dist.get_rank()} did not perform evaluation.")

        logger.info(f"Evaluation done at step {current_step}.")

    def run(self, output_dir):
        log_df = pd.DataFrame()

        current_step = 0
        while current_step < self.total_optimization_steps:
            for batch_number, batch in enumerate(iter(self.dataloader)):
                self.value.train()
                self.policy.train()

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                tokenized_batch = self.collator(batch)

                policy_loss, value_loss = self.model_optimizer.calculate_loss(
                    tokenized_batch
                )

                value_loss.backward()
                policy_loss.backward()

                self.policy_optimizer.step()
                self.value_optimizer.step()

                current_step += 1

                logger.info(
                    f"Step {current_step}, Batch {batch_number + 1}: "
                    f"Policy Loss: {policy_loss.item():.4f}, "
                    f"Value Loss: {value_loss.item():.4f}"
                )

                if dist.get_rank() == 0:
                    step_log_df = pd.DataFrame.from_dict(
                        {
                            "step": [current_step],
                            "policy_loss": [policy_loss.item()],
                            "value_loss": [value_loss.item()],
                        }
                    )

                    log_df = pd.concat([log_df, step_log_df])
                    log_df.to_csv(f"{output_dir}/dro_trainer_log.csv", index=False)
                dist.barrier

                if (current_step % self.check_point_freq == 0) and (current_step > 0):

                    if self.config["trainer"].get("save_models", True):
                        self.save_models(output_dir, current_step)

                    # Run online evaluation if configured
                    if self.config["trainer"].get("evaluate_during_training", False):

                        self.run_evaluation(output_dir, current_step)

                if current_step >= self.total_optimization_steps:
                    break

        # Final save after training completes
        self.policy.module.save(output_dir / "models/final")

        return log_df
