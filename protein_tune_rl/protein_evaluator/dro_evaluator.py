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


class DROEvaluator(Evaluator):
    def __init__(self, config):
        assert dist.get_world_size() == 1
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config["evaluator"]["batch_size"]
        self.model_name = self.config["evaluator"]["model_name"]

        self.dataset = create_dataset(
            name=self.config['dataset']['name'],
            data_directory=self.config['dataset']['data_directory'],
            chain=self.config["dataset"]["chain"],
            region=self.config["dataset"]["region"],
        )

        self.dataloader = create_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

        self.tokenizer = create_tokenizer(
            name=self.config['tokenizer']['name'],
            tokenizer_config=self.config['tokenizer']['tokenizer_config'],
        )

        self.collator = create_collator(
            name=self.config['collator']['name'],
            model_name='gpt2',
            tokenizer=self.tokenizer,
        )

        self.policy = create_model(
            name="iglm",
            hf_config=self.config['policy_model']['dir'],
            vocab_size=self.tokenizer.vocab_size,
        ).to(self.device)

        self.metric_function = []
        for metric in self.config['metric']['name']:
            self.metric_function.append(create_metric(name=metric)())

    def generate(self, starting_tokens, num_to_generate=1, top_p=1, temperature=1):
        # Set to remove duplicates
        sampled_sequences = 0

        while sampled_sequences < num_to_generate:
            seq = self.policy.model.generate(
                starting_tokens.unsqueeze(0),
                max_length=150,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=2,
                forced_eos_token_id=2,
                bad_words_ids=[
                    [0],
                    [1],
                    [3],
                    [4],
                    [25],
                    [26],
                    [27],
                    [28],
                    [29],
                    [30],
                    [31],
                    [32],
                ],
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            ).detach()

            seq = seq[0]  # Squeeze out batch   dimension

            decoded_sequence = [
                self.tokenizer.tokenizer.convert_ids_to_tokens([next_token][0])
                for next_token in seq.tolist()
            ][3:-1]
            infilled_sequence = "".join(
                decoded_sequence[decoded_sequence.index("[SEP]") + 1 :]
            )
            decoded_sequence = "".join(
                decoded_sequence[: decoded_sequence.index("[SEP]") - 1]
            )
            decoded_sequence = decoded_sequence.replace("[MASK]", infilled_sequence)

            sampled_sequences += 1

        return decoded_sequence, infilled_sequence

    def run(self, output_dir):

        eval_df = pd.DataFrame()
        scores, generated_sequences, heavy_chains, light_chains = [], [], [], []

        for batch_number, batch in enumerate(iter(self.dataloader)):
            self.policy.eval()

            tokenized_batch = self.collator(batch, eval=True)

            for idx, sequence in enumerate(
                tokenized_batch['input_ids'].to(self.device)
            ):
                sampled_sequence, sampled_tokens = self.generate(sequence)
                chains = {"L": batch["LC"][idx], "H": sampled_sequence}
                # score the sequence under some eval function (SASA)
                try:
                    score = [
                        metric_function(chains)
                        for metric_function in self.metric_function
                    ]
                except Exception:
                    score = None

                logger.info(
                    f"{batch_number}, seq {sampled_sequence}; infilled seq {sampled_tokens}; score {score}"
                )

                scores.append(score)
                generated_sequences.append(sampled_tokens)
                heavy_chains.append(sampled_sequence)
                light_chains.append(batch["LC"][idx])

        eval_df['completion'] = generated_sequences
        eval_df['HC'] = heavy_chains
        eval_df['LC'] = light_chains
        for idx, metric in enumerate(self.config['metric']['name']):
            eval_df[str(metric)] = [metric_score[idx] for metric_score in scores]

        eval_df.to_csv(f"{output_dir}/{self.model_name}_eval.csv")

        return eval_df
