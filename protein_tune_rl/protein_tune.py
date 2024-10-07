import click
import torch
import numpy as np
import json
from datetime import datetime

from protein_tune_rl.trainer import create_trainer


class ProteinTune:

    def __init__(self, config, id=None):

        

        # read the config file
        with open(config) as f:
            config = json.load(f)

        self.config = config
        if id is not None:
            self.config["scale"]["id"] = id        

        try:
            self.trainer = create_trainer(config['trainer']['name'])(self.config)
        except:
            raise ValueError("------- INITIALIZING TRAINER FAILED --------")

        print("------- INITIALIZED TRAINER --------")

    def train(self, n_run):
        print("------- TRAINER : START --------")
        log = self.trainer.run()
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
def experiment(config_file, runs, id=0):
    print("======= RUNNING ProteinTune EXPERIMENT =======")
    print(f"======= TOTAL RUNS: {runs} =======\n \n")
    for run in range(runs):
        print(f"------- RUN {run} --------")
        torch.manual_seed(run)
        np.random.seed(run)
        qd = ProteinTune(config_file)
        qd.train(run)

    print(" ======= COMPLETED EXPERIMENT =======")


if __name__ == "__main__":
    experiment()
