import os
import pickle
import sys
import torch
import pandas as pd
import torch.distributed as dist


def gather_dataframes(local_df, device, group=None):
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
    tensor = torch.ByteTensor(list(serialized)).to(device)

    # Gather sizes first
    local_size = torch.tensor([tensor.numel()], device=device)
    sizes = [
        torch.tensor([0], device=device) for _ in range(dist.get_world_size(group))
    ]
    dist.all_gather(sizes, local_size, group=group)

    # Pad tensor to max size
    max_size = max(s.item() for s in sizes)
    padded = torch.cat(
        [
            tensor,
            torch.zeros(max_size - tensor.numel(), dtype=torch.uint8, device=device),
        ]
    )

    # Gather all padded tensors
    gathered = [
        torch.empty(max_size, dtype=torch.uint8, device=device)
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


def compute_logp(model, state, action):
    model_out = model(**state)

    logits = model_out.logits[:, -action.shape[-1] - 1 : -1, :]
    logp_mask = state["attention_mask"][:, -action.shape[-1] - 1 : -1]

    all_logps = torch.log_softmax(logits, dim=-1)
    logps = torch.gather(all_logps, dim=-1, index=action.unsqueeze(2)).squeeze(2)

    logps *= logp_mask
    return logps.sum(-1)


def check_pdb(fname):
    """
    Check a PDB (Protein Data Bank) file for specific attributes.

    This function reads a PDB file and checks for the presence of 'END' and 'REMARK'
    lines, indicating the end of the file and additional remarks, respectively.
    Not all structure prediction tools use 'REMARK', but 'END' must always be there.

    Args:
        fname (str): The path to the PDB file to be checked.

    Returns:
        Tuple[bool, bool]: A tuple containing two boolean values.
            - The first value indicates whether the PDB file contains 'END' lines.
            - The second value indicates whether the PDB file contains 'REMARK' lines.
    """
    has_end, has_remark = False, False
    with open(fname, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('END'):
                has_end = True
            if line.startswith('REMARK'):
                has_remark = True
    return has_end, has_remark


class HidePrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def reduce_mean_across_processes(arg0):
    dist.all_reduce(arg0, dist.ReduceOp.SUM)
    arg0 /= dist.get_world_size()
    return arg0


def compute_mean_across_processes(arg0):
    result = arg0.mean()
    return reduce_mean_across_processes(result)


def normalize_across_processes(arg0):
    mean = compute_mean_across_processes(arg0)
    variance = torch.square(arg0.norm(p=2)) / len(arg0) - mean**2
    variance = reduce_mean_across_processes(variance)
    std = torch.sqrt(variance)
    return (arg0 - mean) / (std + 1e-10)
