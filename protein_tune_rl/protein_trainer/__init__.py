import torch


def create_trainer(name):

    if name not in ["ppo", "test", "dro", "online_rl_trainer"]:
        raise ValueError(f"Trainer {name} not supported")

    if name == "test":
        from protein_tune_rl.protein_trainer.test_trainer import TestTrainer

        return TestTrainer

    if name == "dro":
        from protein_tune_rl.protein_trainer.dro_trainer import DROTrainer

        return DROTrainer

    if name == "online_rl_trainer":
        from protein_tune_rl.protein_trainer.online_rl_trainer import \
            OnlineRLTrainer

        return OnlineRLTrainer


def create_optimizer(name):

    if name not in ["adam", "sgd", "adafactor"]:
        raise ValueError(f"Optimizer {name} not supported")

    if name == "adam":
        return torch.optim.Adam
    if name == "sgd":
        return torch.optim.SGD
    if name == "adafactor":
        return torch.optim.Adafactor
