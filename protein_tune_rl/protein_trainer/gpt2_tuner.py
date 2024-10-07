from protein_tuning_rl.protein_tuner.protein_trainer import ProteinTrainer

from protein_tuning_rl.models import create_model
from protein_tuning_rl.dataset import create_dataset
from protein_tuning_rl.dataloader import create_dataloader


from copy import deepcopy
import numpy as np
import torch

import warnings

warnings.filterwarnings("ignore")


class GPT2Trainer(ProteinTrainer):
    def __init__(self, config):
        self.config = config

        self.iterations = self.config["trainer"]["iterations"]
        self.batch_size = self.config["trainer"]["batch_size"]
        self.dataset = create_dataset(self.config['dataset']['data_directory'])
        self.dataloader = create_dataloader(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.model = create_model()

    def run(self):

        

        return None
