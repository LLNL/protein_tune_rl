import pandas as pd
from torch.utils.data import Dataset


class InfillingDataset(Dataset):
    def __init__(self, data_directory, chain, region):
        self.data = pd.read_csv(data_directory)
        self.chain = chain
        self.region = region

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        return {
            "prompts": self.data[self.chain].iloc[idx],
            "region" : self.data[self.region].iloc[idx],
            "LC": self.data.LC.iloc[idx],
        }


