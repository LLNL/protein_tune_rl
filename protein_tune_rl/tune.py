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
        self.config = self._load_config(config)
        self.mode = mode
        self._validate_mode()
        self.fixed_output_dir = self.config.pop('fixed_experiment_directory', False)
        self.exp_output_dir = Path(self.config['experiment_directory'])

        try:
            if self.mode == "tune":
                self._setup_tune()
            elif self.mode == "eval":
                self._setup_eval()
            self._finalize_output_dir()
        except Exception as e:
            raise logger.error(
                f"Failed to initialize ProteinTuneRL. Error : {e}"
            ) from e

        if dist.get_rank() == 0:
            self._save_config()

        logger.info("Initialized ProteinTuneRL")

    def _load_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"Loaded config file: {config_path}")
        return config

    def _validate_mode(self):
        if self.mode not in ["tune", "eval"]:
            raise ValueError(
                f"Mode {self.mode} is not supported. Use 'tune' or 'eval'."
            )

    def _setup_tune(self):
        exp_output_dir = self._build_tune_output_dir()
        self.protein_tuner = create_trainer(self.config['trainer']['name'])(self.config)
        if dist.get_rank() == 0 and not self.fixed_output_dir:
            self.exp_output_dir /= exp_output_dir

    def _setup_eval(self):
        exp_output_dir = self._build_eval_output_dir()
        self.protein_tuner = create_evaluator(self.config['evaluator']['name'])(
            self.config
        )
        if dist.get_rank() == 0 and not self.fixed_output_dir:
            self.exp_output_dir /= exp_output_dir

    def _build_tune_output_dir(self):
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
        else:
            tau = None

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
        return exp_output_dir

    def _build_eval_output_dir(self):
        return (
            "eval/"
            + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
            + '_'
            + self.config['evaluator']['name']
        )

    def _finalize_output_dir(self):
        if dist.get_rank() == 0:
            self.exp_output_dir.mkdir(parents=True, exist_ok=True)

    def _save_config(self):
        with open(self.exp_output_dir / 'config.json', "w") as outfile:
            json.dump(self.config, outfile)

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

    logger.set_rank(rank)
    logger.info("Running ProteinTuneRL experiment")

    if torch.cuda.is_available():
        _set_device_and_print_device_info(rank)
    else:
        logger.info(f"Rank {rank} ➜ CUDA not available; using CPU")

    ws = dist.get_world_size()
    logger.info(f"Rank {rank} ➜ World size (processes) = {ws}")

    for run in range(runs):
        logger.info(f"Run {run + 1}/{runs} - Rank {rank} - Mode: {mode}")
        torch.manual_seed(run)
        np.random.seed(run)
        ProteinTuneRL(config_file, mode).tune()

    logger.info("Completed experiment")

    dist.destroy_process_group()


def _set_device_and_print_device_info(rank):
    """Set the CUDA device for the current process and log device information."""
    device_id = rank % torch.cuda.device_count()
    # Set device for this process will be used by all CUDA operations
    # Without this, each process may try to use the same GPU
    torch.cuda.set_device(device_id)
    # local_gpus = torch.cuda.device_count()
    per_proc_device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(per_proc_device)
    logger.info(f"Rank {rank} ➜ Using CUDA device cuda:{per_proc_device} ({name})")
    # logger.info(f"Rank {rank} ➜ Local visible GPUs = {local_gpus}")


@click.command()
@click.option(
    "-cf",
    "--config-file",
    type=str,
    default=None,
    required=True,
    help="Path to the experiment configuration JSON file.",
)
@click.option(
    "-r",
    "--runs",
    type=int,
    default=1,
    help="Number of repeated runs per process. Useful for statistical averaging or testing robustness.",
)
@click.option(
    "-mode",
    "--mode",
    type=click.Choice(["tune", "eval"], case_sensitive=False),
    default="tune",
    help="Execution mode: 'tune' to train an agent, 'eval' to run an evaluation loop.",
)
@click.option(
    "-np",
    "--num-procs",
    type=int,
    default=-1,
    help="Number of parallel processes to launch (usually maps to GPUs). Use -1 to auto-detect GPU count.",
)
def main(config_file, runs, mode, num_procs):
    # Try multi-node first — safe to fall back if not present
    jsm_rank = os.environ.get("JSM_NAMESPACE_RANK")
    jsm_size = os.environ.get("JSM_NAMESPACE_SIZE")
    if jsm_rank is not None and jsm_size is not None:
        logger.info("Running in multi-node mode (e.g. launched via jsrun)")
        rank = int(jsm_rank)
        num_procs = int(jsm_size)
        experiment(rank, config_file, runs, mode, num_procs)
    else:
        logger.info("Running in single-node mode (e.g. launched via mp.spawn)")
        os.environ["MASTER_ADDR"] = "localhost"
        if num_procs == -1:
            num_procs = torch.cuda.device_count()
            logger.info(
                f"Auto-detected {num_procs} GPUs. Using all available GPUs for parallel processing."
            )
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
