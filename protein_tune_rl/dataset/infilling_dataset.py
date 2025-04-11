import pandas as pd
from torch.utils.data import Dataset


class InfillingDataset(Dataset):
    def __init__(self, data_directory):
        self.data = pd.read_csv(data_directory)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx]
