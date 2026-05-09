# CSDP - Common Sleep Data Pipeline

This repository contains functions and classes for preprocessing and data-loading PSG data, which can then be used for training/predicting with the U-Sleep and LSeqSleepNet models.

There are three submodules:
- **csdp_datastore**: Preprocessing of PSG datasets into HDF5 format. If you already have access to compatible HDF5 files (for example from ERDA), you can ignore this submodule.
- **csdp_pipeline**: PyTorch-based data loading of the preprocessed HDF5 files from the datastore submodule.
- **csdp_training**: PyTorch Lightning based module of U-Sleep and LSeqSleepNet to be used for training, validation, test and simple predictions.

Check out https://gitlab.au.dk/tech_ear-eeg/sleep-code/csdp-demonstration for installation guide and demo scripts.

---

## Installation

### Standard install

```bash
# 1. Install PyTorch for your hardware first.
#    This avoids pip pulling the wrong (CPU-only) wheel when resolving dependencies.

# CUDA 12.8 (most modern NVIDIA GPUs):
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.4 (older NVIDIA clusters):
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cu124

# ROCm 6.2 (AMD — e.g. LUMI supercomputer):
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/rocm6.2

# CPU-only (preprocessing nodes, testing):
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu

# 2. Install the package
pip install -e .

# 3. Install private dependencies (requires GitLab access to gitlab.au.dk)
pip install git+https://gitlab.au.dk/tech_ear-eeg/ml_architectures.git@main
pip install git+https://gitlab.au.dk/tech_ear-eeg/sleep-code/sleep_preprocessing_pipeline.git
pip install git+https://gitlab.au.dk/tech_ear-eeg/sleep-code/sleep_dataset_class.git
```

### Development install

Includes pytest, coverage, and ruff for linting:

```bash
pip install -e ".[dev]"
pip install pre-commit && pre-commit install
```

---

## Running the tests

The test suite requires no data files, no GPU, and no network access.
121 tests run in under 5 seconds.

```bash
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov --cov-report=term-missing
```

Two tests (`test_import_usleep_lightning`, `test_import_usleep_factory`) are skipped
unless `ml_architectures` is installed. With it installed, all 123 tests pass.

See [docs/testing.md](docs/testing.md) for a full description of every test file and what each test covers.

---

## CI/CD

See [docs/cicd.md](docs/cicd.md) for the full CI/CD documentation including:
- GitHub Actions CI (lint, build, test matrix on Python 3.11 + 3.12)
- Docker image build and push to GHCR (on version tags)
- Pre-commit hooks
- HPC deployment script (`scripts/deploy_hpc.sh`)
