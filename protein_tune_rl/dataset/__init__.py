
def create_dataset(name, data_directory):
    if name == "sequence":
        from protein_tune_rl.dataset.sequence_dataset import SequenceDataset
        return SequenceDataset(data_directory=data_directory)

    else:
        return NotImplementedError