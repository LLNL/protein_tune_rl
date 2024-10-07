
def create_dataset(name, data_dir):
    if name == "sequence":
        from protein_tune_rl.dataset.sequence_dataset import SequenceDataset
        return SequenceDataset(data_directory=data_dir)

    else:
        return NotImplementedError