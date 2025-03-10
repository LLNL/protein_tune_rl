import torch


def create_trainer(name):
    if name == "test":
        from protein_tune_rl.protein_trainer.test_trainer import TestTrainer

        return TestTrainer

    if name == "dro":
        from protein_tune_rl.protein_trainer.dro_trainer import DROTrainer

        return DROTrainer


def create_optimizer(name):
    if name == "adam":
        return torch.optim.Adam
    if name == "sgd":
        return torch.optim.SGD
    if name == "adafactor":
        return torch.optim.Adafactor
    if name is None:
        raise ValueError(f"Optimizer {name} not supported")
