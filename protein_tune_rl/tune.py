import click
import torch
import numpy as np
import json
from datetime import datetime

from protein_tune_rl.protein_trainer import create_trainer


class ProteinTuneRL:

    def __init__(self, config, mode):        
        
        # read the config file
        with open(config) as f:
            self.config = json.load(f)

        try:
            if mode == "tune":
                self.protein_tuner = create_trainer(self.config['trainer']['name'])(self.config)
            if mode == "evaluate":
                raise NotImplementedError
        except:
            raise ValueError("------- INITIALIZING TRAINER FAILED --------")

        print("------- INITIALIZED TRAINER --------")

    def train(self, n_run):
        print("------- TRAINER : START --------")
        log = self.protein_tuner.run()
        print("------- TRAINER : FINISHED --------")

        if log is not None:        
            if self.config['trainer']["save_results"]:
                log.write_to_disk(self.config, self.config['trainer']['name'], n_run)


#######################################################################
#                       RUN EXPERIMENT
#######################################################################


@click.command()
@click.option("-cf", "--config-file", type=str, default=None)
@click.option("-r", "--runs", type=int, default=1)
@click.option("-mode", "--mode", type=str, default="tune")

def experiment(config_file, runs, mode):
    print("======= RUNNING ProteinTuneRL EXPERIMENT =======")
    print(f"======= TOTAL RUNS: {runs} =======\n \n")
    for run in range(runs):
        print(f"------- RUN {run} --------")
        torch.manual_seed(run)
        np.random.seed(run)
        ProteinTuneRL(config_file, mode).train(run)

    print(" ======= COMPLETED EXPERIMENT =======")


if __name__ == "__main__":
    experiment()
