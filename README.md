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

## Introduction

Welcome to the `ProteinTuneRL` repository! ProteinTuneRL is an innovative framework that applies reinforcement learning (RL) to the challenging task of protein design. Traditional protein engineering often grapples with the vastness of the protein sequence space, making it difficult to pinpoint sequences that exhibit optimal stability, activity, or specificity. By integrating state-of-the-art generative models with tailored RL algorithms, ProteinTuneRL provides a robust and systematic approach to fine-tuning these models, steering them toward generating protein sequences with desired properties.

## Theoretical Background

ProteinTuneRL allows users to fine-tune a generative model $\pi_{\theta}(y|x)$ to generate protein sequences with desired properties. This is achieved by maximizing the following objective:

$$
\max_{\pi_{\theta}} \left(
\underbrace{\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\theta}(y|x)} \left[ r(x, y) \right]}_{\substack{\text{generate sequences} \\ \text{with favorable properties}}} - \beta\, \underbrace{\mathbb{D}_{\text{KL}} \left[ \pi_{\theta}(y|x) \,\|\, \pi_{\text{ref}}(y|x) \right]}_{\substack{\text{maintain the likelihood} \\ \text{of the original model}}}
\right)
$$
wehre $r(x, y)$ is the reward function that evaluates the quality of the generated sequence $y$ given the input $x$, $\mathcal{D}$ is the dataset of input sequences $x$, and $\pi_{\text{ref}}$ is a reference model that the fine-tuned model $\pi_{\theta}$ should not deviate too far from.

### Online Learning

In the online learning setting, the dataset $\mathcal{D}$ is not fixed, and the model $\pi_{\theta}$ is updated in an iterative manner. The objective can be rewritten as:

$$
\max_{\pi_{\theta}} \left(
\mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_{\theta}(y|x)} \left[ 
r(x, y) - \beta\, \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)}
\right]
\right).
$$

We provide two algorithms, `reinforce` and `ppo`, to optimize the above objective. The `reinforce` algorithm is a simple policy gradient method, while the `ppo` algorithm is a more advanced method that uses a clipped surrogate objective to stabilize the training process.

### Offline Learning

For many protein design tasks, the dataset $\mathcal{D} = \{ x_i, y_i, r_i \}_{i=1}^N$ is fixed and can be used to pre-train the model $\pi_{\theta}$. For example, experimental data can be used to train a model that generates sequences with high stability. We provide `dro`, an implementation of the [Offline Regularised Reinforcement Learning for Large Language Models Alignment](https://arxiv.org/abs/2405.19107) algorithm to optimize the following derivation of the above objective:

$$
\mathcal{L}_{\text{DRO}} = \frac{1}{2} \mathbb{E}_{ \{x,y,r \} \sim \mathcal{D} }
\left[
  \left( r - V_{\varphi}(x) - \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} \right)^2 
\right]
$$

where $V(x)$ is a value function that estimates the quality of the input $x$.

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

