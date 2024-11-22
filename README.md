# ProteinTuneRL

ProteinTuneRL is a minimal implementation of reinforcement learning fine-tuning for protein design.

## Basic installation

```bash
pip install -e '.'
```

### For development

```bash
pip install -e '.[dev]'
```
For convenience, you can use automatic editing tools like `black` and `isort`:

```bash
black -S -t py39 protein_tune_rl
flake8 --ignore=E501,E203,W503 protein_tune_rl
isort protein_tune_rl # isort will automatically sort imports in the correct order
```
