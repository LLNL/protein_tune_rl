import json
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import torch

from protein_tune_rl import logger
from protein_tune_rl.protein_evaluator import create_evaluator
from protein_tune_rl.protein_trainer import create_trainer


class ProteinTuneRL:
    def __init__(self, config, mode):
        # read the config file
        with open(config) as f:
            self.config = json.load(f)

        try:
            if mode == "tune":
                self.exp_output_dir = (
                    self.config['experiment_directory']
                    + self.config['trainer']['name']
                    + f"/{self.config['dataset']['reward']}"
                    + f"/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_"
                    + self.config['trainer']['name']
                    + '_steps_'
                    + str(self.config['trainer']['total_optimization_steps'])
                    + '_bs_'
                    + str(self.config['trainer']['batch_size'])
                    + '_lr_'
                    + str(self.config['trainer']['learning_rate'])
                )

                Path(self.exp_output_dir).mkdir(parents=True, exist_ok=True)
                self.protein_tuner = create_trainer(self.config['trainer']['name'])(
                    self.config
                )

            if mode == "eval":
                self.exp_output_dir = (
                    self.config['experiment_directory']
                    + "/eval/"
                    + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
                    + '_'
                    + self.config['evaluator']['name']
                )

                Path(self.exp_output_dir).mkdir(parents=True, exist_ok=True)
                self.protein_tuner = create_evaluator(self.config['evaluator']['name'])(
                    self.config
                )
        except Exception as e:
            raise logger.error(f"Error: {e}") from e

        with open(f"{self.exp_output_dir}/config.json", "w") as outfile:
            json.dump(self.config, outfile)

        logger.info("Initialized ProtTuneRL")

    def tune(self, n_run):
        logger.info("Starting ProtTuneRL")
        __ = self.protein_tuner.run(self.exp_output_dir)
        logger.info("Finished ProtTuneRL")


#######################################################################
#                       RUN EXPERIMENT
#######################################################################


@click.command()
@click.option("-cf", "--config-file", type=str, default=None)
@click.option("-r", "--runs", type=int, default=1)
@click.option("-mode", "--mode", type=str, default="tune")
def experiment(config_file, runs, mode):

    logger.info("Running ProteinTuneRL Experiment")
    logger.info(f"Total runs: {runs}")
    for run in range(runs):
        logger.info(f"Run {run}")
        torch.manual_seed(run)
        np.random.seed(run)
        ProteinTuneRL(config_file, mode).tune(run)

    logger.info("Completed experiment")


if __name__ == "__main__":
    experiment()
