from typing import List

import pandas as pd
import torch
import torch.distributed as dist

from protein_tune_rl.protein_evaluator.evaluator import Evaluator
from protein_tune_rl.protein_trainer.online_rl_trainer import OnlineRLSampler


class OnlineRLEvaluator(Evaluator, OnlineRLSampler):
    def __init__(self, config):
        Evaluator.__init__(self, config)
        OnlineRLSampler.__init__(self, config)

    def _gather(self, tensor: torch.tensor) -> List[torch.tensor]:
        tensor = tensor.to(self.device)
        if dist.get_rank() == 0:
            all_tensors = [
                torch.zeros_like(tensor, device=self.device)
                for _ in range(dist.get_world_size())
            ]
        else:
            all_tensors = None
        dist.gather(tensor, all_tensors)
        dist.barrier()
        out = []
        if dist.get_rank() == 0:
            for t in all_tensors:
                out.extend(t.tolist())
        return out

    def run(self, exp_output_dir):
        scores = {name: [] for name in self.metrics}
        for init_seqs in self.dataloader:
            with torch.no_grad():
                sampled_seqs, _, _ = self._sample_batch(self.model, init_seqs)

            for name, metric in self.metrics.items():
                score = torch.zeros(len(sampled_seqs))
                for i, seq in enumerate(sampled_seqs):
                    chains = {
                        "L": init_seqs["LC"][i],
                        "H": seq,
                        "seq_pre_mask": init_seqs["seq_pre_mask"][i],
                        "seq_post_mask": init_seqs["seq_post_mask"][i],
                    }
                    score[i] = metric(chains)
                scores[name].extend(self._gather(score))

        if dist.get_rank() == 0:
            df = pd.DataFrame(scores)
            df.to_csv(exp_output_dir / "eval_scores.csv")
