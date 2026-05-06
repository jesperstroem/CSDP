# CSDP - Common Sleep Data Pipeline

This repository contains functions and classes for preprocessing and data-loading PSG data, which can then be used for training/predicting with the U-Sleep model.

There are three submodules:
- **csdp_datastore**: Preprocessing of PSG datasets into HDF5 format. If you already have access to compatible HDF5 files (for example from ERDA), you can ignore this submodule.
- **csdp_pipeline**: PyTorch-based data loading of the preprocessed HDF5 files from the datastore submodule.
- **csdp_training**: PyTorch Lightning based module of U-Sleep to be used for training, validation, test and simple predictions.

Check out https://gitlab.au.dk/tech_ear-eeg/sleep-code/csdp-demonstration for installation guide and demo scripts.

---

## Installation

```bash
# 1. Install CPU torch first (avoids downloading the 2 GB CUDA wheel)
pip install "torch~=2.11.0" --index-url https://download.pytorch.org/whl/cpu

# 2. Install the package
pip install -e .

# 3. Install ml_architectures (required for training — needs GitLab access)
pip install git+https://gitlab.au.dk/tech_ear-eeg/sleep-code/sleep_dataset_class.git
```

For development (includes pytest and ruff):

```bash
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

---

## CI/CD

See [docs/cicd.md](docs/cicd.md) for the full CI/CD documentation including:
- GitHub Actions CI (lint, build, test matrix)
- Docker image build and push to GHCR
- Pre-commit hooks
- HPC deployment script