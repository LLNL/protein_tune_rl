import pickle

import pandas as pd
import torch
import torch.distributed as dist

from protein_tune_rl import logger
from protein_tune_rl.dataloader import create_dataloader
from protein_tune_rl.dataset import create_dataset
from protein_tune_rl.metrics import create_metric
from protein_tune_rl.protein_evaluator.evaluator import Evaluator


class SequenceEvaluator(Evaluator):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = self.config["evaluator"]["batch_size"]
        self.eval_name = self.config["evaluator"]["model_name"]

        self.dataset = create_dataset(
            name=self.config['dataset']['name'],
            data_directory=self.config['dataset']['data_directory'],
            chain=self.config["dataset"]["chain"],
            region=self.config["dataset"]["region"],
        )

        self.dataloader = create_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

        self.metric_function = []
        self.metric_function.extend(
            create_metric(name=metric["name"])(**metric["params"])
            for metric in self.config['metric']
        )

    def run(self, output_dir):

        eval_df = pd.DataFrame()
        scores, heavy_chains, light_chains = [], [], []

        for batch_number, batch in enumerate(iter(self.dataloader)):

            for sequence, LC in zip(batch['prompts'], batch["LC"]):

                chains = {
                    "L": LC,
                    "H": sequence,
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
                    f"rank {dist.get_rank()}; {batch_number}, seq {sequence} ; score {score}"
                )

                scores.append(score)
                heavy_chains.append(sequence)
                light_chains.append(LC)

        eval_df['HC'] = heavy_chains
        eval_df['LC'] = light_chains

        for idx, metric in enumerate(self.config['metric']):
            eval_df[str(metric['name'])] = [
                metric_score[idx] for metric_score in scores
            ]

        final_df = self.gather_dataframes(eval_df)

        if dist.get_rank() == 0:
            final_df.to_csv(f"{output_dir}/{self.eval_name}_eval.csv")

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
