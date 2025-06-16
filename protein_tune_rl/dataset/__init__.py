def create_dataset(name, data_directory, chain=None, region=None, reward=None):
    try:
        if name == "sequence":
            from protein_tune_rl.dataset.sequence_dataset import SequenceDataset

            return SequenceDataset(data_directory=data_directory)

        if name == "dro":
            from protein_tune_rl.dataset.dro_dataset import DRODataset

            return DRODataset(
                data_directory=data_directory, chain=chain, region=region, reward=reward
            )

        if name == "dro_eval":
            from protein_tune_rl.dataset.dro_dataset import DROEvalDataset

            return DROEvalDataset(
                data_directory=data_directory, chain=chain, region=region
            )

        if name == "infilling":
            from protein_tune_rl.dataset.infilling_dataset import InfillingDataset

            return InfillingDataset(
                data_directory=data_directory, chain=chain, region=region
            )

        raise ValueError(f"Unknown dataset name: {name}")

    except Exception as e:
        raise RuntimeError(f"Failed to create dataset '{name}': {e}") from e
