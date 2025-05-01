import pickle

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


class IGLMEvaluator(Evaluator):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config["evaluator"]["batch_size"]
        self.model_name = self.config["evaluator"]["model_name"]

        self.num_to_generate = self.config["generator"]["num_to_generate"]
        self.top_p = self.config["generator"]["top_p"]
        self.temperature = self.config["generator"]["temperature"]
        self.max_length = self.config["generator"]["max_length"]
        self.bad_word_ids = self.config["generator"]["bad_word_ids"]

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

        for metric in self.config['metric']:
            print(metric)
            print(metric["name"])
            print(metric["params"])
            #print(self.config['metric']['name'][metric]["params"])

        self.metric_function = []
        self.metric_function.extend(
            create_metric(name=metric["name"])(**metric["params"]) for metric in self.config['metric']
        )

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
            ][3:-1]
            
            decoded_infill = "".join(
                decoded_sequence[decoded_sequence.index("[SEP]") + 1 :]
            )

            decoded_sequence = "".join(
                decoded_sequence[: decoded_sequence.index("[SEP]") - 1]
            )
            decoded_sequence = decoded_sequence.replace("[MASK]", decoded_infill)
            
            decoded_sequences.append(decoded_sequence)
            decoded_infills.append(decoded_infill)

        return decoded_sequences, decoded_infills

    def run(self, output_dir):

        eval_df = pd.DataFrame()
        scores, generated_sequences, heavy_chains, light_chains = [], [], [], []

        for batch_number, batch in enumerate(iter(self.dataloader)):
            self.policy.eval()

            tokenized_batch = self.collator(batch, eval=True)

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


                for full_sampled_sequence, infilled_sequence in zip(full_sampled_sequences, infilled_sequences):

                    chains = {
                            "L": batch["LC"][idx], 
                            "H": full_sampled_sequence, 
                            "seq_pre_mask" : tokenized_batch["seq_pre_mask"], 
                            "seq_post_mask" : tokenized_batch["seq_post_mask"]
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
                        f"rank {dist.get_rank()}; {batch_number}, seq {full_sampled_sequence}; infilled seq {infilled_sequence}; score {score}"
                    )

                    scores.append(score)
                    generated_sequences.append(infilled_sequence)
                    heavy_chains.append(full_sampled_sequence)
                    light_chains.append(batch["LC"][idx])

        eval_df['completion'] = generated_sequences
        eval_df['HC'] = heavy_chains
        eval_df['LC'] = light_chains
        for idx, metric in enumerate(self.config['metric']['name']):
            eval_df[str(metric)] = [metric_score[idx] for metric_score in scores]

        final_df = self.gather_dataframes(eval_df)

        if dist.get_rank() == 0:
            final_df.to_csv(f"{output_dir}/{self.model_name}_eval.csv")

        return final_df

    def gather_dataframes(self, local_df, group=None):
        """
        Gather pandas DataFrames from all processes and combine them on rank 0.

        Args:
            local_df (pd.DataFrame): Local DataFrame on each process.
            group (optional): Torch distributed process group.

        Returns:
            pd.DataFrame on rank 0, None elsewhere.
        """

        # Serialize the DataFrame using pickle
        serialized = pickle.dumps(local_df)
        tensor = torch.ByteTensor(list(serialized)).to(self.device)

        # Gather sizes first
        local_size = torch.tensor([tensor.numel()], device=self.device)
        sizes = [
            torch.tensor([0], device=self.device)
            for _ in range(dist.get_world_size(group))
        ]
        dist.all_gather(sizes, local_size, group=group)

        # Pad tensor to max size
        max_size = max(s.item() for s in sizes)
        padded = torch.cat(
            [
                tensor,
                torch.zeros(
                    max_size - tensor.numel(), dtype=torch.uint8, device=self.device
                ),
            ]
        )

        # Gather all padded tensors
        gathered = [
            torch.empty(max_size, dtype=torch.uint8, device=self.device)
            for _ in range(dist.get_world_size(group))
        ]
        dist.all_gather(gathered, padded, group=group)

        if dist.get_rank(group) == 0:
            dfs = []
            for t, s in zip(gathered, sizes):
                raw = bytes(t[: s.item()].tolist())
                df = pickle.loads(raw)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

        return None
