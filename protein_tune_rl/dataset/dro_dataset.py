import pandas as pd
from torch.utils.data import Dataset

from protein_tune_rl.dataset.infilling_dataset import InfillingDataset


class DRODataset(InfillingDataset):
    def __init__(self, data_directory, chain, region, reward):
        super().__init__(data_directory, chain, region)
        self.reward = reward


    def __getitem__(self, idx):

        return {
            "prompts": self.data[self.chain].iloc[idx],
            "completions": self.data[self.region].iloc[idx],
            "rewards": float(self.data[self.reward].iloc[idx]),
            "LC": self.data.LC.iloc[idx],
        }


class DROEvalDataset(InfillingDataset):
    def __init__(self, data_directory, chain, region):
        super().__init__(data_directory, chain, region)

    def __getitem__(self, idx):

        return {
            "prompts": self.data[self.chain].iloc[idx],
            "completions": self.data[self.region].iloc[idx],
            "LC": self.data.LC.iloc[idx],
        }
