from distutils.core import setup


required = [
    "numpy>=1.19",
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
    "PyYAML",
    "requests==2.24.0",
    "sacremoses==0.0.43",
    #"tokenizers",
    #"torchvision",
    "transformers",
    #"urllib3",
]

extras = {
    "dev": [
        "pytest",
        "black==22.8.0",
        "flake8==5.0.4",
        "isort",
    ],
    "folding": [
        "igfold"
    ]
}
extras['all'] = list({item for group in extras.values() for item in group})

setup(
    name='protein_tune_rl',
    version='1.0dev',
    description='Optimization for Protein Language Models',
    author='LLNL',
    packages=['protein_tune_rl'],
    install_requires=required,
    extras_require=extras,
)
