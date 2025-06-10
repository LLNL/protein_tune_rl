import json
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from protein_tune_rl import logger
from protein_tune_rl.protein_evaluator import create_evaluator
from protein_tune_rl.protein_trainer import create_trainer

warnings.filterwarnings("ignore")


class ProteinTuneRL:
    def __init__(self, config, mode):
        self.exp_output_dir = None
        tau = None

        # read the config file
        with open(config) as f:
            self.config = json.load(f)

        logger.info(f"Loaded config file: {config}")
        if mode not in ["tune", "eval"]:
            raise ValueError(f"Mode {mode} is not supported. Use 'tune' or 'eval'.")

        self.exp_output_dir = Path(self.config['experiment_directory'])
        fixed_output_dir = self.config.pop('fixed_experiment_directory', False)

        try:
            if mode == "tune":
                if not fixed_output_dir:
                    if 'reward' in self.config['dataset']:
                        reward = self.config['dataset']['reward']
                    else:
                        reward = self.config['metric'][0]['name']

                    if 'learning_rate' in self.config['trainer']:
                        lr = self.config['trainer']['learning_rate']
                    else:
                        lr = self.config['optimizer']['learning_rate']
                    if 'tau' in self.config['trainer']:
                        tau = self.config['trainer']['tau']
                    elif 'tau' in self.config['optimizer']:
                        tau = self.config['optimizer']['tau']

                    exp_output_dir = (
                        self.config['trainer']['name']
                        + f"/{reward}"
                        + f'/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_'
                        + self.config['trainer']['name']
                        + '_steps_'
                        + str(self.config['trainer']['total_optimization_steps'])
                        + '_bs_'
                        + str(self.config['trainer']['batch_size'])
                        + '_lr_'
                        + str(lr)
                    )

                    if tau is not None:
                        exp_output_dir += f'_tau_{str(tau)}'

                self.protein_tuner = create_trainer(self.config['trainer']['name'])(
                    self.config
                )

            if mode == "eval":
                if not fixed_output_dir:
                    exp_output_dir = (
                        "eval/"
                        + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
                        + '_'
                        + self.config['evaluator']['name']
                    )

                self.protein_tuner = create_evaluator(self.config['evaluator']['name'])(
                    self.config
                )

            if dist.get_rank() == 0:
                if not fixed_output_dir:
                    self.exp_output_dir /= exp_output_dir
                self.exp_output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise logger.error(f"Failed to initialize ProteinTuneRL: Error {e}") from e

        if dist.get_rank() == 0:
            with open(self.exp_output_dir / 'config.json', "w") as outfile:
                json.dump(self.config, outfile)

        logger.info("Initialized ProteinTuneRL")

    def tune(self):
        logger.info("Starting ProteinTuneRL")
        self.protein_tuner.run(self.exp_output_dir)
        logger.info("Finished ProteinTuneRL")


#######################################################################
#                       RUN EXPERIMENT
#######################################################################


def experiment(rank, config_file, runs, mode, num_procs):
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "23358"

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(
        backend, rank=rank, world_size=num_procs, timeout=timedelta(seconds=60)
    )

    if torch.cuda.is_available():
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)

    logger.set_rank(rank)
    logger.info("Running ProteinTuneRL experiment")

    for run in range(runs):
        logger.info(f"Run {run + 1}/{runs} - Rank {rank} - Mode: {mode}")
        torch.manual_seed(run)
        np.random.seed(run)
        ProteinTuneRL(config_file, mode).tune()

    logger.info("Completed experiment")

    dist.destroy_process_group()


@click.command()
@click.option("-cf", "--config-file", type=str, default=None)
@click.option("-r", "--runs", type=int, default=1)
@click.option("-mode", "--mode", type=str, default="tune")
@click.option("-np", "--num-procs", type=int, default=-1)
def main(config_file, runs, mode, num_procs):
    try:  # multi-node
        rank = int(os.environ['JSM_NAMESPACE_RANK'])
        num_procs = int(os.environ['JSM_NAMESPACE_SIZE'])
        experiment(rank, config_file, runs, mode, num_procs)
    except KeyError:  # single-node
        os.environ["MASTER_ADDR"] = "localhost"
        if num_procs == -1:
            num_procs = torch.cuda.device_count()
        mp.spawn(
            experiment,
            args=(
                config_file,
                runs,
                mode,
                num_procs,
            ),
            nprocs=num_procs,
            join=True,
        )


if __name__ == "__main__":
    main()
