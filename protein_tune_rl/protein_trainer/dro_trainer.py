import pandas as pd
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

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
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            #collate_fn=self.collator,
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
        )

        # Initialize the optimizer.
        self.optimizer_class = create_optimizer(self.config["trainer"]["optimizer"])
        self.policy_optimizer = self.optimizer_class(
            self.policy.parameters(), lr=self.learning_rate
        )
        self.value_optimizer = self.optimizer_class(
            self.value.parameters(), lr=self.learning_rate
        )

    def run(self, output_dir):
        log_df = pd.DataFrame()

        current_step = 0
        while current_step < self.total_optimization_steps:
            for batch_number, batch in enumerate(iter(self.dataloader)):
                self.value.train()
                self.policy.train()

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()

                # print(batch)
                # print("")

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
                    f"step {current_step}, batch: {batch_number+1}; policy loss: {policy_loss}; value loss {value_loss}"
                )

                step_log_df = pd.DataFrame.from_dict(
                    {
                        "step": [current_step],
                        "policy_loss": [policy_loss.item()],
                        "value_loss": [value_loss.item()],
                    }
                )

                log_df = pd.concat([log_df, step_log_df])
                log_df.to_csv(f"{output_dir}/dro_log.csv")

                if (current_step % self.check_point_freq == 0) and (current_step > 0):
                    # save policy network to disk
                    torch.save(
                        self.policy.state_dict(),
                        f"{output_dir}/policy_model_{current_step}.bin",
                    )

                    # save value network to disk
                    torch.save(
                        self.value.state_dict(),
                        f"{output_dir}/value_model_{current_step}.bin",
                    )

                if current_step >= self.total_optimization_steps:
                    break

        return log_df
