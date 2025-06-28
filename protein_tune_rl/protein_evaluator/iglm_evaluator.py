import pandas as pd
import torch
import torch.distributed as dist

from protein_tune_rl import logger
from protein_tune_rl.collator import create_collator
from protein_tune_rl.dataloader import create_dataloader
from protein_tune_rl.dataset import create_dataset
from protein_tune_rl.metrics import create_metric
from protein_tune_rl.models import create_model
from protein_tune_rl.protein_evaluator.evaluator import Evaluator
from protein_tune_rl.tokenizer import create_tokenizer
from protein_tune_rl.util.util import gather_dataframes


class IGLMEvaluator(Evaluator):
    def __init__(self, config, policy_model=None):
        """
        Initializes the IGLM Evaluator with the provided configuration and policy model.

        Args:
            config (dict): Configuration dictionary containing parameters for evaluation.
            policy_model (optional): Pre-trained policy model to be used for evaluation. If None, a new model will be created.
        """
        super().__init__(config)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config["evaluator"]["batch_size"]
        if self.batch_size != 1:
            raise ValueError("Only batch size of 1 currently supported for evaluation.")

        self.model_name = self.config["evaluator"]["model_name"]
        self.num_to_generate = self.config["generator"]["num_to_generate"]
        self.top_p = self.config["generator"]["top_p"]
        self.temperature = self.config["generator"]["temperature"]
        self.max_length = self.config["generator"]["max_length"]
        self.bad_word_ids = self.config["generator"]["bad_word_ids"]

        self.dataset = create_dataset(
            name=self.config['dataset_eval']['name'],
            data_directory=self.config['dataset_eval']['data_directory'],
            chain=self.config['dataset_eval']["chain"],
            region=self.config['dataset_eval']["region"],
        )

        self.tokenizer = create_tokenizer(
            name=self.config['tokenizer']['name'],
            tokenizer_config=self.config['tokenizer']['tokenizer_config'],
            padding_side=self.config['tokenizer']['padding_side'],
        )

        self.collator = create_collator(
            name=self.config['collator_eval']['name'],
            tokenizer=self.tokenizer,
            eval=True,
        )

        self.dataloader = create_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        # If external policy model is provided, use it
        if policy_model is not None:
            self.policy = policy_model
        else:
            self.policy = create_model(
                name="iglm",
                hf_config=self.config['policy_model']['dir'],
                vocab_size=self.tokenizer.vocab_size,
            ).to(self.device)

        self.metric_function = []
        self.metric_function.extend(
            create_metric(name=metric["name"])(**metric["params"])
            for metric in self.config['metric']
        )

        # Which metrics use generated sequences?
        self.metric_use_generated = [
            metric_cfg.get("use_generated", True)
            for metric_cfg in self.config['metric']
        ]

    def update_policy(self, new_policy):
        """
        Replace the current policy model with a new one (e.g., from training).
        Useful for online evaluation to avoid re-instantiating the evaluator.
        """
        logger.info("Updating IGLM model in evaluator")
        self.policy = new_policy

        # Update model reference inside each metric that has update_model()
        for metric in self.metric_function:
            if hasattr(metric, "update_model"):
                metric.update_model(new_policy)

    def generate(
        self,
        starting_tokens,
        num_to_generate,
        top_p,
        temperature,
        max_length,
        bad_word_ids,
    ):

        decoded_sequences, decoded_infills = [], []
        for __ in range(num_to_generate):
            tokens = self.policy.model.generate(
                starting_tokens.unsqueeze(0),
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=2,
                forced_eos_token_id=2,
                bad_words_ids=bad_word_ids,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            ).detach()

            tokens = tokens[0]  # Squeeze out batch   dimension

            # decode sequence ids for IgLM
            decoded_sequence = [
                self.tokenizer.tokenizer.convert_ids_to_tokens([next_token][0])
                for next_token in tokens.tolist()
            ][2:-1]

            decoded_infill = "".join(
                decoded_sequence[decoded_sequence.index("[SEP]") + 1 :]
            )

            decoded_sequence = "".join(
                decoded_sequence[: decoded_sequence.index("[SEP]")]
            )
            decoded_sequence = decoded_sequence.replace("[MASK]", decoded_infill)

            decoded_sequences.append(decoded_sequence)
            decoded_infills.append(decoded_infill)

        return decoded_sequences, decoded_infills

    def run(self, output_dir):

        eval_df = pd.DataFrame()
        prompts, scores, generated_sequences, heavy_chains, light_chains = (
            [],
            [],
            [],
            [],
            [],
        )

        for batch_number, batch in enumerate(iter(self.dataloader)):
            self.policy.eval()

            tokenized_batch = self.collator(batch)

            for idx, sequence in enumerate(
                tokenized_batch['input_ids'].to(self.device)
            ):
                full_sampled_sequences, infilled_sequences = self.generate(
                    sequence,
                    self.num_to_generate,
                    self.top_p,
                    self.temperature,
                    self.max_length,
                    self.bad_word_ids,
                )

                for full_sampled_sequence, infilled_sequence in zip(
                    full_sampled_sequences, infilled_sequences
                ):

                    chains = {
                        "L": batch["LC"][idx],
                        "H": full_sampled_sequence,
                        "seq_pre_mask": tokenized_batch["seq_pre_mask"],
                        "seq_post_mask": tokenized_batch["seq_post_mask"],
                    }

                    # score the sequence under some eval function (SASA)
                    try:
                        score = [
                            metric_function(chains)
                            for metric_function in self.metric_function
                        ]
                    except Exception:
                        score = None

                    logger.info(
                        f"Rank {dist.get_rank()}; "
                        f"Batch {batch_number + 1}, "
                        f"Sampled Sequence: {full_sampled_sequence}, "
                        f"Infilling: {infilled_sequence}, "
                        f"Score: {score}"
                    )

                    scores.append(score)
                    generated_sequences.append(infilled_sequence)
                    heavy_chains.append(full_sampled_sequence)
                    light_chains.append(batch["LC"][idx])
                    prompts.append(
                        tokenized_batch["seq_pre_mask"][0]
                        + "[MASK]"
                        + tokenized_batch["seq_post_mask"][0]
                    )

        eval_df['completion'] = generated_sequences
        eval_df['HC'] = heavy_chains
        eval_df['LC'] = light_chains
        eval_df['prompts'] = prompts

        for idx, metric in enumerate(self.config['metric']):
            eval_df[str(metric['name'])] = [
                metric_score[idx] for metric_score in scores
            ]

        final_df = gather_dataframes(eval_df, device=self.device)

        if dist.get_rank() == 0:
            final_df.to_csv(f"{output_dir}/{self.model_name}_evaluator_log.csv")

        return final_df
