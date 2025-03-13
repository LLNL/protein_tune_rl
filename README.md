# ProteinTuneRL

ProteinTuneRL is a minimal implementation of reinforcement learning fine-tuning for protein design.

## Basic installation

- Basic installation can be done by running the following command: 
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

