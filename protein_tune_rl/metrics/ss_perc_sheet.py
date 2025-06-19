from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PercBetaSheet:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, chains):
        X = ProteinAnalysis(str(chains['H']) + str(chains['L']))

        return (X.secondary_structure_fraction()[2] - self.mean) / self.std
