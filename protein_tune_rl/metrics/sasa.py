import tempfile

from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley


class SASA:
    def __init__(self):
        from igfold import IgFoldRunner
        from igfold.refine.pyrosetta_ref import init_pyrosetta

        self.parser = PDBParser(QUIET=1)
        self.sr = ShrakeRupley()

        init_pyrosetta()
        self.igfold = IgFoldRunner()

    def __call__(self, chains):

        pdb_fn = tempfile.NamedTemporaryFile(suffix='.pdb').name
        self.igfold.fold(
            pdb_fn,  # Output PDB file
            sequences=chains,  # Antibody sequences
            do_refine=True,  # Refine the antibody structure with PyRosetta
            do_renum=False,  # Renumber predicted antibody structure (Chothia)
        )

        struct = self.parser.get_structure("dummy", pdb_fn)

        # try:
        self.sr.compute(struct, level="S")
        # labels["avg_rd"] = self.avg_res_depth(struct)
        # except Exception as e:
        #     pass

        return struct.sasa
