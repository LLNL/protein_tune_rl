import os
from typing import Dict

import torch
import torch.distributed as dist
from igfold import IgFoldRunner

from protein_tune_rl.util.util import HidePrints, check_pdb


class StructureBasedMetric:
    def __init__(self, folding_tool: str, options: Dict = {}):
        assert folding_tool.lower() == "igfold", "Currently only IgFold is supported!"
        self.options = options

        if options["do_refine"] and not options["use_openmm"]:
            from igfold.refine.pyrosetta_ref import init_pyrosetta

            init_pyrosetta()

        num_models = self.options.pop("num_models", 1)
        with HidePrints():  # hide prints from igfold
            self.igfold = IgFoldRunner(num_models=num_models)

        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")

        self.igfold.antiberty.model.to(device)
        for model in self.igfold.models:
            model.to(device)

        rank = dist.get_rank()
        self.workspace = f"folding_workspace/rank{rank}/"
        os.makedirs(self.workspace, exist_ok=True)

        self.count = 0

    def _fold(self, chains: Dict, name: str):
        num_tries = 0
        output_pdb_file = self.workspace + name + ".pdb"

        while True:
            try:
                with HidePrints():  # hide prints from igfold
                    out = self.igfold.fold(
                        output_pdb_file,  # Output PDB file
                        sequences=chains,  # Antibody sequences
                        **self.options,
                        # do_refine=True, # Refine the antibody structure with PyRosetta
                        # use_openmm=False, # Use OpenMM for refinement
                        # do_renum=False, # Renumber predicted antibody structure (Chothia)
                    )

                pdb_has_end, _ = check_pdb(output_pdb_file)
            except Exception as e:
                print(f"Error: {e}")
                pdb_has_end = False

            num_tries += 1

            if pdb_has_end:
                break
            if num_tries >= 3:
                print(f"Warning: folding of {output_pdb_file[:-4]} failed!")
                out = None
                break

            os.remove(output_pdb_file)
            os.remove(self.workspace + name + ".fasta")

        self.count += 1

        return output_pdb_file, out
