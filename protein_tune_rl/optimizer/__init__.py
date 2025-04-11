def create_optimizer(name, model, **kwargs):
    if name == "reinforce":
        from protein_tune_rl.optimizer.reinforce import Reinforce

        return Reinforce(model=model, **kwargs)
    elif name == "ppo":
        from protein_tune_rl.optimizer.ppo import PPO

        return PPO(model=model, **kwargs)
