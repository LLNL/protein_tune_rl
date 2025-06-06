import os
import sys
from collections import defaultdict
import torch

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
