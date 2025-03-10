from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PercBetaSheet:
    def __call__(self, chains):
        X = ProteinAnalysis(str(chains['H']))

        return X.secondary_structure_fraction()[2]
