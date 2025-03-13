import os
from typing import Dict

from protein_tune_rl.metrics.structure import StructureBasedMetric


class FoldingConfidence(StructureBasedMetric):
    def __init__(self, folding_tool: str = "igfold", options={}):
        super().__init__(folding_tool, options)

    def __call__(self, chains: Dict):
        name = "fold" + str(self.count)
        output_pdb_file, out = self._fold(chains, name)

        os.remove(output_pdb_file)
        os.remove(self.workspace + name + ".fasta")

        # prmsd: Predicted RMSD for each residue's N, CA, C, CB atoms (dim: 1, L, 4)
        res_rmsd = out.prmsd.square().mean(dim=-1).sqrt().squeeze(0)
        return 1.0 / (1.0 + res_rmsd.mean().cpu().numpy())

    def __repr__(self):
        return "1 / (1 + predicted RMSD)"
