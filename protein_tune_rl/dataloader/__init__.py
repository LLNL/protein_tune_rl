import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from protein_tune_rl import logger


def create_dataloader(dataset, batch_size, shuffle=True, collate_fn=None):

    if batch_size > len(dataset):
        logger.info(
            f"Batch size {batch_size} is larger than dataset size {len(dataset)}. Adjusting batch size to dataset size."
        )
        batch_size = len(dataset)

    # num_replicas is the number of processes in the distributed setup
    # If the dataset is smaller than the number of processes, we should not drop data
    # to ensure all processes have data to work with.
    # drop_last is a boolean indicating whether to drop the last incomplete batch.
    num_reps = dist.get_world_size()
    drop = len(dataset) > num_reps

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        drop_last=drop,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        drop_last=True,
    )
