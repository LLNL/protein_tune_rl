from abc import ABC
import numpy as np
import os
import random
import torch
import torch.distributed as dist

from protein_tune_rl import logger


class Trainer(ABC):
    def __init__(self, config):
        self.config = config

        # Set device (use GPU if available)
        if torch.cuda.is_available():
            self.device_ids = [torch.cuda.current_device()]
            self.device = torch.device("cuda", self.device_ids[0])
        else:
            self.device_ids = None
            self.device = torch.device("cpu")

        self.ckpt = None
        if "checkpoint" in config:
            ckpt_path = config["checkpoint"]
            self.ckpt = self._load_checkpoint(ckpt_path, self.device)

    def run(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def _log_dataset_info(self, dataloader, logger):
        dl = dataloader
        world = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        sampler = getattr(dl, "sampler", None)
        per_rank_samples = len(sampler) if sampler is not None else len(dl.dataset)
        per_rank_batches = len(dl)
        logger.info(
            f"Per-rank: {per_rank_samples} samples → {per_rank_batches} batches "
            f"(batch size={dl.batch_size}, drop_last={dl.drop_last}); "
            f"Global: world_size={world}, effective batch size={dl.batch_size * world}, "
            f"batches/epoch={per_rank_batches * world}."
        )

    def _unwrap_ddp_model(self, model):
        """Unwrap DDP to get the underlying model module."""
        return (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )

    def _should_checkpoint(self, current_step, check_point_freq):
        return (current_step % check_point_freq == 0) and (current_step > 0)

    def _gather_cuda_rng_states(self):
        wz = dist.get_world_size()
        rank = dist.get_rank()
        loc_s = torch.cuda.get_rng_state_all()[self.device_ids[0]].to(self.device)
        out = [torch.empty_like(loc_s, device=self.device) for _ in range(wz)]
        dist.gather(loc_s, gather_list=(None if rank else out), dst=0)
        return out if rank == 0 else None

    def _save_checkpoint(self, ckpt_dir, tag, step):
        state = {
            "step": step,
            "policy": self._unwrap_ddp_model(self.policy).state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": self._gather_cuda_rng_states(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        if hasattr(self, "value"):
            state["value"] = self._unwrap_ddp_model(self.value).state_dict()
            state["value_optimizer"] = self.value_optimizer.state_dict()

        final_path = os.path.join(ckpt_dir, f"{tag}_s{step}.ckpt")
        if dist.get_rank() == 0:
            try:
                tmp_final = f"{final_path}.tmp"
                torch.save(state, tmp_final)
                os.replace(tmp_final, final_path)
            except Exception as e:
                logger.error(f"Checkpoint save FAILED: {e}")
                raise
        dist.barrier()

    def _load_checkpoint(self, ckpt_path, device):
        if device.type == "cpu":
            map_location = "cpu"
        else:
            map_location = {"cuda:0": f"cuda:{device.index}"}
        ckpt = torch.load(ckpt_path, map_location, weights_only=False)

        torch.set_rng_state(ckpt["rng_state"]["torch"])
        state = ckpt["rng_state"]["cuda"][dist.get_rank()].cpu()
        torch.cuda.set_rng_state(state, device=self.device)
        np.random.set_state(ckpt["rng_state"]["numpy"])
        random.setstate(ckpt["rng_state"]["python"])

        return ckpt

    def _maybe_load_state_dict(self, obj, state_dict):
        if self.ckpt and self.ckpt.get(state_dict) is not None:
            self._unwrap_ddp_model(obj).load_state_dict(self.ckpt[state_dict])
