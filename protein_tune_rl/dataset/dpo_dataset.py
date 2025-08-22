import pandas as pd
from torch.utils.data import Dataset


class DPODataset(Dataset):
    """Dataset for DPO training, containing contexts and preference pairs (y+ and y-)."""

    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the dataset file (CSV, JSONL, or Parquet) containing preference pairs.
        """
        # Load data file into DataFrame
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".json") or file_path.endswith(".jsonl"):
            df = pd.read_json(file_path, lines=True)
        elif file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format for DPO dataset: {file_path}")

        # Expected columns: at least context (prompt) and two completions.
        # Identify relevant column names (case-insensitive fallback if needed).
        # Assuming standard column names as in example:
        self.hc_masked = (
            df["HC_Masked"].astype(str).tolist()
        )  # heavy chain with [MASK] token in sequence
        self.y_pos = df["HCDR3_winning"].astype(str).tolist()
        self.y_neg = df["HCDR3_losing"].astype(str).tolist()
        # Optional columns
        self.LC_seq = df["LC"].astype(str).tolist() if "LC" in df.columns else None
        self.fitness_w = (
            df["fitness_winning"].tolist() if "fitness_winning" in df.columns else None
        )
        self.fitness_l = (
            df["fitness_losing"].tolist() if "fitness_losing" in df.columns else None
        )

    def __len__(self):
        return len(self.hc_masked)

    def __getitem__(self, idx):
        # Base output with required fields
        sample = {
            "__row_idx__": idx,
            "prompt": self.hc_masked[idx],
            "completion_pos": self.y_pos[idx],
            "completion_neg": self.y_neg[idx],
        }
        # Include optional fields for completeness (not used in training loss, but may be logged)
        if self.LC_seq is not None:
            sample["LC"] = self.LC_seq[idx]
        if self.fitness_w is not None and self.fitness_l is not None:
            sample["fitness_winning"] = self.fitness_w[idx]
            sample["fitness_losing"] = self.fitness_l[idx]
        return sample
