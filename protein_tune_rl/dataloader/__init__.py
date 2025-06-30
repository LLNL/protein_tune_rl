import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from protein_tune_rl import logger


def create_dataloader(dataset, batch_size, shuffle=True, collate_fn=None):

    if batch_size > len(dataset):
        logger.warning(
            f"Batch size {batch_size} is larger than dataset size {len(dataset)}. Adjusting batch size to dataset size."
        )
        batch_size = len(dataset)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        drop_last=True,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=True,
    )
