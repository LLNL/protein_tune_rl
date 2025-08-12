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

        logger.info(f"Models saved at step {current_step} to {output_dir}.")

    def run_evaluation(self, output_dir, current_step):
        """Run evaluation at the current training step."""
        logger.info(f"Running evaluation at step {current_step}...")

        # If DDP-wrapped, pass .module (unwrap the DDP model)
        eval_policy = self._unwrap_ddp_model(self.policy)

        # Update evaluator with the current policy model
        self.evaluator.update_policy(eval_policy)

        with torch.no_grad():
            eval_df = self.evaluator.run_with_ground_truth()

        if dist.get_rank() == 0 and eval_df is not None:
            eval_df.to_csv(
                f"{output_dir}/evaluation_results_step_{current_step}.csv",
                index=False,
            )
        dist.barrier()

        logger.info(f"Evaluation done at step {current_step}.")

    def run(self, output_dir):
        """Run the DRO Trainer for the specified number of optimization steps."""
        log_df = pd.DataFrame()

        self._log_dataset_info()

        current_step = 0
        while current_step < self.total_optimization_steps:
            for batch_number, batch in enumerate(iter(self.dataloader)):
                current_step = self._train_step(batch, current_step)
                self._log_step(log_df, output_dir, current_step, batch_number)
                dist.barrier()

                if self._should_checkpoint(current_step):
                    self._maybe_save_models(output_dir, current_step)
                    self._maybe_run_evaluation(output_dir, current_step)

                if current_step >= self.total_optimization_steps:
                    break

        self._final_save(output_dir)
        return log_df

    def _log_dataset_info(self):
        logger.info(
            f"Each process will handle {len(self.dataloader.dataset) // dist.get_world_size()} samples."
        )
        logger.info(
            f"The load of each process is broken down into {len(self.dataloader)} batches."
        )

    def _train_step(self, batch, current_step):
        self.value.train()
        self.policy.train()

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        tokenized_batch = self.collator(batch)

        policy_loss, value_loss = self.model_optimizer.calculate_loss(tokenized_batch)

        value_loss.backward()
        policy_loss.backward()

        self.policy_optimizer.step()
        self.value_optimizer.step()

        logger.info(
            f"Step {current_step + 1}, Batch: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}"
        )

        self._last_policy_loss = policy_loss
        self._last_value_loss = value_loss

        return current_step + 1

    def _log_step(self, log_df, output_dir, current_step, batch_number):
        if dist.get_rank() == 0:
            step_log_df = pd.DataFrame.from_dict(
                {
                    "step": [current_step],
                    "policy_loss": [self._last_policy_loss.item()],
                    "value_loss": [self._last_value_loss.item()],
                }
            )
            log_df = pd.concat([log_df, step_log_df])
            log_df.to_csv(f"{output_dir}/dro_trainer_log.csv", index=False)

    def _should_checkpoint(self, current_step):
        return (current_step % self.check_point_freq == 0) and (current_step > 0)

    def _maybe_save_models(self, output_dir, current_step):
        if self.config["trainer"].get("save_models", True):
            if dist.get_rank() == 0:
                self.save_models(output_dir, current_step)
            dist.barrier()

    def _maybe_run_evaluation(self, output_dir, current_step):
        if self.config["trainer"].get("evaluate_during_training", False):
            self.run_evaluation(output_dir, current_step)

    def _final_save(self, output_dir):
        self.policy.module.save(output_dir / "models/final")
