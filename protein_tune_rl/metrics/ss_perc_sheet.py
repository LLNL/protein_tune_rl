from typing import Dict

from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PercBetaSheet:
    def __call__(self, chains):
        X = ProteinAnalysis(str(chains['H']) + str(chains['L']))

        return X.secondary_structure_fraction()[2]
