from abc import ABC
import torch.distributed as dist


class Trainer(ABC):
    def __init__(self, config):
        self.config = config

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

    def _should_checkpoint(self, current_step, check_point_freq):
        return (current_step % check_point_freq == 0) and (current_step > 0)
