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

## Basic installation

Basic installation can be done by running the following command: 
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.'
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
