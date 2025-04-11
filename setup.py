from setuptools import setup

setup(
    use_scm_version={"write_to": "protein_tune_rl/_version.py"},
    setup_requires=['setuptools_scm']
)
