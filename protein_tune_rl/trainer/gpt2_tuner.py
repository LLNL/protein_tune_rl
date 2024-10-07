from protein_tuning_rl.trainer.trainer import Trainer

from protein_tuning_rl.models import create_model


from copy import deepcopy
import numpy as np
import torch

import warnings

warnings.filterwarnings("ignore")


class GPT2Trainer(Trainer):
    def __init__(self, config):
        self.config = config

        self.iterations = self.config["trainer"]["iterations"]
        self.batch_size = self.config["trainer"]["batch_size"]

        self.model = create_model()

    def run(self):

        

        return None
