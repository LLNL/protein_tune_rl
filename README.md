# ProteinTuneRL

ProteinTuneRL is a minimal implementation of reinforcement learning fine-tuning for protein design.

## Basic installation

- Basic installation can be done by running the following command: 
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e '.'
```

### For development

- Install the package in development mode with the following command:
```bash
pip install -e '.[dev]'
```

- For convenience, you can use automatic editing tools like `black`, `flake8`, and `isort` to format your code. You can run the following commands to format your code:   

```bash
black -S -t py39 protein_tune_rl
flake8 --ignore=E501,E203,W503 protein_tune_rl
isort protein_tune_rl # isort will automatically sort imports in the correct order
```
