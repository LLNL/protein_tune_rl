<div align="left">
  <h2>
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/proteintunerl-logo-name-dark.png" width="350">
    <source media="(prefers-color-scheme: light)" srcset="images/proteintunerl-logo-name-light.png" width="350">
    <img alt="protlib-designer" src="images/proteintunerl-logo-name-light.png" width="350">
    </picture>
  </h2>
</div>

![Status](https://img.shields.io/badge/Status-Active-green.svg)
![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![CI](https://github.com/LLNL/protlib-designer/actions/workflows/ci.yml/badge.svg)

## Introduction

Welcome to the `ProteinTuneRL` repository! ProteinTuneRL is an innovative framework that applies reinforcement learning (RL) to the challenging task of protein design. Traditional protein engineering often grapples with the vastness of the protein sequence space, making it difficult to pinpoint sequences that exhibit optimal stability, activity, or specificity. By integrating state-of-the-art generative models with tailored RL algorithms, ProteinTuneRL provides a robust and systematic approach to fine-tuning these models, steering them toward generating protein sequences with desired properties.

## Theoretical Background

The main objective of this project is :

$$
\max_{\pi_{\theta}} \left(
\underbrace{\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\theta}(y|x)} \left[ r(x, y) \right]}_{\substack{\text{generate sequences} \\ \text{with favorable properties}}} - \beta\, \underbrace{\mathbb{D}_{\text{KL}} \left[ \pi_{\theta}(y|x) \,\|\, \pi_{\text{ref}}(y|x) \right]}_{\substack{\text{maintain the likelihood} \\ \text{of the original model}}}
\right)
$$

Recall that the definition of the KL divergence is:
$$
\mathbb{D}_{\text{KL}} \left[ \pi_{\theta}(y|x) \,\|\, \pi_{\text{ref}}(y|x) \right] = \mathbb{E}_{y \sim \pi_{\theta}(y|x)} \left[ \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} \right].
$$

Therefore, the objective can be rewritten as:
$$
\max_{\pi_{\theta}} \left(
\mathbb{E}_{x,y} \left[ 
r(x, y) - \beta\, \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}
\right]
\right).
$$

where:


### PPO Loss

In our case, $\pi_{\text{ref}}$ acts as a fixed baseline (or pre-trained model). We can define the ratio:
$$
r_t(\theta) = \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}.
$$
Then the reward part of the loss can be incorporated using the PPO clipping mechanism:
$$
L_{\text{reward}}(\theta) = \mathbb{E}_{x,y}\left[ \min\left( r_t(\theta)\, \hat{A}(x,y),\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, \hat{A}(x,y) \right) \right].
$$

Putting these components together, the overall loss is:
$$
L(\theta) = -\left\{ \mathbb{E}_{x,y \sim \pi_{\theta_{\text{old}}}} \left[\min\left( r_t(\theta) \hat{A}(x,y),\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}(x,y) \right) \right] - \beta\, \mathbb{D}_{\text{KL}}\left[\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\right] \right\}.
$$
A few notes on this formulation:
- **Advantage Estimation:** $\hat{A}(x,y)$ can be computed using techniques like Generalized Advantage Estimation (GAE), comparing observed rewards to a baseline (often estimated by a value function).
- **Clipping Mechanism:** The clipping term $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$ restricts the update so that the new policy does not deviate too far from $\pi_{\text{ref}}$ in a single update.
- **Adaptive Penalty:** In some implementations, the coefficient $\beta$ can be adapted during training to balance between optimizing the reward and maintaining proximity to the reference policy.
- **Additional Terms:** In a full PPO implementation (especially in actor-critic setups), there may be additional terms such as a value function loss and an entropy bonus for exploration.

## Basic installation

Basic installation can be done by running the following command: 
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.'
```

## Installation on Lassen

- Follow the steps in [Nikoli Dryden's note](https://lc.llnl.gov/confluence/display/~dryden1/PyTorch+2.5+from+source+on+Lassen), up to the section `Set up your environment for future use`. Adjust the paths/environment names as needed. Remark: if you would like to use the `openmm` refine feature in `IgFold`, choose python 3.9 instead of python 3.11 when setting up the conda environment.
- Suppose the conda environment created in the previous step is named `ProteinTuneRL`, activate the environment:
```bash
conda activate ProteinTuneRL
```
- (Optional) install `openmm` (note that the latest version on ppc64le is 7.6, as opposed to 7.7 recommended by `IgFold`):
```bash
conda install openmm
```
- Install `ProteinTuneRL`:
```bash
pip install -e '.'
```
or if folding signal is needed:
```bash
pip install -e '.'[folding]
```

## Running on Lassen

- On a single node (4 GPUs by default):
```bash
python tune.py -cf config/ft_iglm_folding.json
```
- On multiple nodes (the following example assumes the number of nodes is 2, which is specified after `-N`)
```bash
export MASTER_ADDR=$(jsrun --nrs 1 -r 1 hostname)
lrun -T4 -N2 python tune.py -cf configs/ft_iglm_folding.json
```

### For development

Install the package in development mode with the following command:
```bash
pip install -e '.[dev]'
```

For convenience, you can use automatic editing tools like `black`, `flake8`, and `isort` to format your code. You can run the following commands to format your code:   

```bash
black -S -t py39 protein_tune_rl
flake8 --ignore=E501,E203,W503 protein_tune_rl
isort protein_tune_rl # isort will automatically sort imports in the correct order
```

