from protein_tune_rl.protein_trainer.trainer import Trainer

from protein_tune_rl.models import create_model
from protein_tune_rl.dataset import create_dataset
from protein_tune_rl.dataloader import create_dataloader
from protein_tune_rl.tokenizer import create_tokenizer


from copy import deepcopy
import numpy as np
import torch

import warnings

warnings.filterwarnings("ignore")


class TestTrainer(Trainer):
    def __init__(self, config):
        self.config = config

        self.epochs = self.config["trainer"]["epochs"]
        self.batch_size = self.config["trainer"]["batch_size"]

        self.dataset = create_dataset(name=self.config['dataset']['name'],
                                      data_directory=self.config['dataset']['data_directory'])
        
        self.dataloader = create_dataloader(self.dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=True)
        
        self.tokenizer = create_tokenizer(name=self.config['tokenizer']['name'],
                                          tokenizer_config=self.config['tokenizer']['tokenizer_config'])
        
        self.model = create_model(name="gpt2", vocab_size=self.tokenizer.vocab_size)

    def run(self):

        for e in range(self.epochs):            
            print("=== Epoch", e, "===")
            for batch_number, batch_seqeuence in enumerate(iter(self.dataloader)): 
                input_ids = self.tokenizer(batch_seqeuence)
                output = self.model(input_ids)

                print("Batch number", batch_number, "; Output shape", output[0].shape)
            
            print("")
        

        return None
