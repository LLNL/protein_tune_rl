from distutils.core import setup


required = [
    "numpy>=1.19",
    "torch",
    "sympy",
    "pandas",
    "scikit-learn",
    "click",
    "pathos",
    "seaborn",
    "progress",
    "tqdm",
    "commentjson",
    "pydantic",
    "PyYAML",
    "biopython",
    "chardet==3.0.4",
    "filelock==3.0.12",
    "idna==2.10",
    "PyYAML==5.4.1",
    "requests==2.24.0",
    "sacremoses==0.0.43",
    "tokenizers",
    "torchvision",
    "transformers",
    "urllib3",
]

extras = {
    "docs": [
        "numpydoc>=1.1.0",
        "sphinx>=4.2.0",
        "sphinx-rtd-theme==1.0.0",
        "sphinx-copybutton==0.4.0",
        "sphinx-multiversion==0.2.4",
    ],
    "dev": [
        "pytest",
        "black==22.8.0",
        "flake8==5.0.4",
    ],
}
extras['all'] = list(set([item for group in extras.values() for item in group]))

setup(
    name='protein_tune_rl',
    version='1.0dev',
    description='Optimization for Protein Language Models',
    author='LLNL',
    packages=['protein_tune_rl'],
    install_requires=required,
    extras_require=extras,
)
