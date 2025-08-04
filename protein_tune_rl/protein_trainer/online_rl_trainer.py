from functools import cached_property

import pandas as pd
import torch
import torch.distributed as dist
from torch.distributions.categorical import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP

from protein_tune_rl import logger
from protein_tune_rl.collator import create_collator
from protein_tune_rl.dataloader import create_dataloader
from protein_tune_rl.dataset import create_dataset
from protein_tune_rl.metrics import create_metric
from protein_tune_rl.models import create_model
from protein_tune_rl.optimizer import create_optimizer
from protein_tune_rl.protein_trainer.trainer import Trainer
from protein_tune_rl.tokenizer import create_tokenizer
from protein_tune_rl.util.util import compute_logp


class KLPenalty:
    def __init__(self, ref_model, weight, target, device=None):
        self.ref_model = ref_model
        self.beta = weight
        self.K_b = 0.1
        self.target = target

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

    def __call__(self, logp, sequences):
        init_size = sequences["init_size"]
        state = {
            "input_ids": sequences["input_ids"].to(self.device),
            "attention_mask": sequences["attention_mask"].to(self.device),
            "position_ids": sequences["position_ids"].to(self.device),
        }

        action = state["input_ids"][:, init_size:].detach()

        with torch.no_grad():
            ref_logp = compute_logp(self.ref_model, state, action)
        KL = logp.detach() - ref_logp

        # section 2.2 of https://arxiv.org/pdf/1909.08593
        if self.target is not None:
            rel_KL = KL - self.target / self.target
            e_t = torch.clamp(rel_KL, -0.2, 0.2)
            self.beta = self.beta * (1 + self.K_b * e_t)

        return -self.beta * KL


class OnlineRLSampler:
    def __init__(self, config):
        sample_mode = "trainer" if "trainer" in config else "evaluator"
        self.batch_size = config[sample_mode]["batch_size"]
        self.max_length = config[sample_mode]["max_length"]

        self.dataset = create_dataset(
            name=config['dataset']['name'],
            data_directory=config['dataset']['data_directory'],
            chain=config["dataset"]["chain"],
            region=config["dataset"]["region"],
        )

        tokenizer = create_tokenizer(**config['tokenizer'])
        self.tokenizer = tokenizer

        self.collator = create_collator(name="infilling", tokenizer=self.tokenizer)

        self.dataloader = create_dataloader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if torch.cuda.is_available():
            self.device_ids = [torch.cuda.current_device()]
            self.device = torch.device("cuda", self.device_ids[0])
        else:
            self.device_ids = None
            self.device = torch.device("cpu")

        self.attn_impl = config['policy_model'].get('attn_implementation', "eager")
        self.model = create_model(
            name="iglm",
            hf_config=config['policy_model']['dir'],
            vocab_size=self.tokenizer.vocab_size,
            attn_implementation=self.attn_impl,
        ).to(self.device)
        self.model.eval()
        self.model = DDP(self.model, device_ids=self.device_ids)

        self.metrics = {}
        for m in config['metric']:
            metric = create_metric(name=m["name"])(**m["params"])
            self.metrics[m["name"]] = metric

        self.stop_token_id = tokenizer.stop_token_id

    @cached_property
    def _constrained_ids(self):
        return self.tokenizer.tokenizer.convert_tokens_to_ids(
            [
                "[PAD]",
                "[UNK]",
                "[SEP]",
                "[MASK]",
                "[CAMEL]",
                "[HUMAN]",
                "[MOUSE]",
                "[RABBIT]",
                "[RAT]",
                "[RHESUS]",
                "[HEAVY]",
                "[LIGHT]",
            ]
        )

    def _update_input(self, input, new_token, finished):
        new_pos = input["position_ids"][:, -1:].clone()
        not_finished = ~finished
        new_pos += 1

        input["input_ids"] = torch.cat((input["input_ids"], new_token[:, None]), 1)
        input["attention_mask"] = torch.cat(
            (input["attention_mask"], not_finished[:, None]), 1
        )
        input["position_ids"] = torch.cat((input["position_ids"], new_pos), 1)

    def _decode(self, model_output, init_sequences):
        infilled_ids = torch.cat(model_output, 1)

        sampled_seq = []
        for i, ids in enumerate(infilled_ids):
            terminal_idx = (ids == self.stop_token_id).nonzero()
            if len(terminal_idx) > 0:
                infilled_seq = self.tokenizer.tokenizer.decode(
                    ids[: terminal_idx[0, 0]]
                )
            else:
                infilled_seq = self.tokenizer.tokenizer.decode(ids)
            infilled_seq = infilled_seq.replace(" ", "")

            seq = (
                init_sequences["seq_pre_mask"][i]
                + infilled_seq
                + init_sequences["seq_post_mask"][i]
            )

            sampled_seq.append(seq)

        return sampled_seq

    def _sample_batch(self, model, init_sequences):
        model_input = {
            "input_ids": init_sequences["input_ids"].to(self.device),
            "attention_mask": init_sequences["attention_mask"].to(self.device),
            "position_ids": init_sequences["position_ids"].to(self.device),
        }
        logp_sum = torch.zeros(len(model_input["input_ids"]), device=self.device)
        entropy_sum = torch.zeros(len(model_input["input_ids"]), device=self.device)
        finished = torch.zeros_like(logp_sum, dtype=torch.bool, device=self.device)
        samples = []

        for key, value in model_input.items():
            model_input[key] = value.to(self.device)

        for _ in range(self.max_length):
            output = model(**model_input)
            logits = output.logits[:, -1, :]
            logits[:, self._constrained_ids] = -float('inf')

            prob_dist = Categorical(logits=logits)
            sample = prob_dist.sample().detach()
            samples.append(sample[:, None])

            logp = prob_dist.log_prob(sample)
            logp_sum += logp * (~finished)

            entropy = prob_dist.entropy()
            entropy_sum += entropy * (~finished)

            just_finished = sample == self.stop_token_id
            finished = torch.logical_or(just_finished, finished)

            finish = finished.all()
            dist.all_reduce(finish, dist.ReduceOp.PRODUCT)

            self._update_input(model_input, sample, finished)

            if finish:
                break

        sampled_seq = self._decode(samples, init_sequences)

        init_sequences["input_ids"] = model_input["input_ids"]
        init_sequences["attention_mask"] = model_input["attention_mask"]
        init_sequences["position_ids"] = model_input["position_ids"]

        return sampled_seq, logp_sum, entropy_sum


class OnlineRLTrainer(Trainer, OnlineRLSampler):
    def __init__(self, config):
        Trainer.__init__(self, config)
        OnlineRLSampler.__init__(self, config)

        self.check_point_freq = config["trainer"]["check_point_freq"]
        self.total_optimization_steps = self.config["trainer"][
            "total_optimization_steps"
        ]

        self.use_KL_penalty = config["KL_penalty"]["weight"] > 0.0
        if self.use_KL_penalty:
            self.ref_model = create_model(
                name="iglm",
                hf_config=config['policy_model']['dir'],
                vocab_size=self.tokenizer.vocab_size,
            ).to(self.device)
            self.ref_model.eval()
            self.ref_model = DDP(self.ref_model, device_ids=self.device_ids)
            self.KL_penalty = KLPenalty(
                self.ref_model, **config["KL_penalty"], device=self.device
            )

        optimizer_name = config["optimizer"].pop("name")
        if self.attn_impl == "eager" and optimizer_name == "reinforce":
            logger.info(
                "Warning: when the optimizer is 'reinforce', policy_model with"
                " attn_implementation='eager' can cause RuntimeError during "
                "gradient computation. Please use a different attn_implementation."
            )

        self.optimizer = create_optimizer(
            name=optimizer_name,
            model=self.model,
            **config["optimizer"],
        )

        assert len(self.metrics) == 1, "only single metric is supported"
        self.metric = list(self.metrics.values())[0]

    def run(self, exp_output_dir):
        batch_size = self.batch_size * dist.get_world_size()

        log_df = pd.DataFrame()
        current_step = 0
        while current_step < self.total_optimization_steps:
            for batch_number, batch in enumerate(iter(self.dataloader)):
                tokenized_batch = self.collator(batch)
                sampled_seqs, logp, entropy = self._sample_batch(
                    self.model, tokenized_batch
                )

                reward = torch.zeros(len(sampled_seqs))
                for i, seq in enumerate(sampled_seqs):
                    chains = {"L": batch["LC"][i], "H": seq}
                    reward[i] = self.metric(chains)
                reward = reward.to(logp.device)

                reward_mean = reward.mean()
                dist.all_reduce(reward_mean, dist.ReduceOp.SUM)
                reward_mean /= dist.get_world_size()

                if self.use_KL_penalty:
                    KL_penalty = self.KL_penalty(logp, tokenized_batch)
                    reward += KL_penalty
                    KL_mean = KL_penalty.mean()
                    dist.all_reduce(KL_mean, dist.ReduceOp.SUM)
                    KL_mean /= dist.get_world_size()
                    logger.info(f"step: {current_step}, KL mean: {KL_mean}")

                self.optimizer.step(reward, reward_mean, logp, entropy, tokenized_batch)

                reward_mean = reward_mean.cpu().numpy()
                logger.info(f"step: {current_step}, reward mean: {reward_mean}")

                current_step += 1

                if dist.get_rank() == 0:
                    if current_step % self.check_point_freq == 0:
                        self.model.module.save(
                            exp_output_dir / f"models/batch{current_step}"
                        )

                    step_log = pd.DataFrame(
                        {
                            "num_samples": [current_step * batch_size],
                            "reward_mean": [reward_mean],
                        }
                    )

                    if self.use_KL_penalty:
                        step_log["KL_mean"] = [
                            KL_mean.cpu().numpy() / -self.KL_penalty.beta
                        ]
                    log_df = pd.concat([log_df, step_log], ignore_index=True)
                    log_df.to_csv(exp_output_dir / "train_log.csv")
                dist.barrier()

                if current_step >= self.total_optimization_steps:
                    break

        if dist.get_rank() == 0:
            self.model.module.save(exp_output_dir / "models/final")
