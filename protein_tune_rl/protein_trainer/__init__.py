
def create_trainer(name):
    if name == "test":
        from protein_tune_rl.protein_trainer.test_trainer import TestTrainer
        return TestTrainer