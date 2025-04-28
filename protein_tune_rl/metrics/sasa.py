import os
from typing import Dict

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley

from protein_tune_rl.metrics.structure import StructureBasedMetric


class SASA(StructureBasedMetric):
    def __init__(self, folding_tool: str, options: Dict = None):
        if options is None:
            options = {"do_renum": False}
        super().__init__(folding_tool, options)

        self.parser = PDBParser(QUIET=1)
        self.sr = ShrakeRupley()

    def __call__(self, chains: Dict):
        name = f"fold{str(self.count)}"
        output_pdb_file, _ = self._fold(chains, name)

        struct = self.parser.get_structure(name, output_pdb_file)
        self.sr.compute(struct, level="S")

        os.remove(output_pdb_file)
        os.remove(self.workspace + name + ".fasta")

        return struct.sasa

    def __repr__(self):
        return "SASA"
