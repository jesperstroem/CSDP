# CSDP Changelog

All changes made to the original CSDP project during the CI/CD and migration work.

---

## Package definition

| File | Change |
|---|---|
| `setup.py` | **Deleted** — replaced by `pyproject.toml` |
| `pyproject.toml` | **Created** — full package declaration: name, version, `requires-python = ">=3.11"`, all dependencies, ruff config, pytest config, coverage config |

Dependency decisions in `pyproject.toml`:
- `torch>=2.7.0` — raised from 2.5 to match `sleep_dataset_class` constraint
- `lightning~=2.6.1` — replaces deprecated `pytorch_lightning`
- `pandas>=2.1.4,<3` — upper bound added for MLflow compatibility
- `mlflow>=2.0` — replaces `neptune>=1.13`
- `torchmetrics`, `mne-bids`, `xlsxwriter` — added as new deps not present in original CSDP

---

## Source code migrations

| File | Change |
|---|---|
| `csdp_training/__init__.py` | Wrapped `USleep_Lightning`, `LSeqSleepNet_Lightning`, `USleep_Factory` imports in `try/except ImportError` so the package imports cleanly without `ml_architectures` |
| `csdp_training/lightning_models/base.py` | `import pytorch_lightning as pl` → `import lightning as pl` |
| `csdp_training/lightning_models/usleep.py` | `import pytorch_lightning as pl` → `import lightning as pl` |
| `csdp_training/lightning_models/factories/lightning_model_factory.py` | `import pytorch_lightning as pl` → `import lightning as pl` |
| `csdp_training/experiments/cv.py` | `import pytorch_lightning as pl` → `import lightning as pl`; `from pytorch_lightning.callbacks` → `from lightning.pytorch.callbacks`; Neptune logger replaced with MLflow (see below) |
| `csdp_training/utility.py` | Comment updated: "Neptune" → "MLflow" |

### Neptune → MLflow migration (`cv.py`)

| Before | After |
|---|---|
| `import neptune` | `import mlflow` |
| `from neptune.utils import stringify_unsupported` | *(removed)* |
| `from lightning.pytorch.loggers import NeptuneLogger` | `from lightning.pytorch.loggers import MLFlowLogger` |
| `neptune_run: neptune.Run \| None = None` parameter | `mlflow_run_id: str \| None = None` parameter |
| `self.neptune_run[f"{split_name}/split_data"] = stringify_unsupported(...)` | `mlflow.MlflowClient().log_dict(run_id=..., dictionary=..., artifact_file=...)` |
| `NeptuneLogger(run=self.neptune_run, prefix=split_name)` | `MLFlowLogger(run_id=self.mlflow_run_id, prefix=split_name)` |

New usage pattern:

```python
import mlflow

mlflow.set_experiment("usleep_cv")

with mlflow.start_run(run_name="my_run") as run:
    experiment = CV_Experiment(
        ...,
        mlflow_run_id=run.info.run_id,
    )
    experiment.run()
```

---

## CI/CD infrastructure

All files below are new.

| File | What it does |
|---|---|
| `.github/workflows/ci.yml` | 3-job GitHub Actions pipeline: lint (ruff), build wheel, pytest matrix on Python 3.11 + 3.12 with CPU torch |
| `.github/workflows/cd.yml` | Docker build → push to GitHub Container Registry on `v*.*.*` version tags |
| `Dockerfile` | Multi-target image: `PYTORCH_CHANNEL` build-arg supports `cu128`, `cu124`, `rocm6.2`, `cpu`; smoke-tests imports at build time |
| `.dockerignore` | Excludes `.git`, data files (`*.hdf5`, `*.parquet`), model weights (`*.ckpt`), `tests/`, `.venv/` |
| `.pre-commit-config.yaml` | ruff lint + format hooks; pre-commit-hooks for YAML/TOML validation, large file guard (500 KB), merge conflict detection, debug statement detection |
| `scripts/deploy_hpc.sh` | SSH pull script for HPC clusters (LUMI/PRIME): checks commits behind origin, `git pull --ff-only`, optional `sbatch` submission |

---

## Tests

All files below are new. Total: **123 tests, all passing.**

| File | Tests | What is covered |
|---|---|---|
| `tests/conftest.py` | 2 fixtures | `synthetic_batch_128hz`, `synthetic_batch_100hz` shared across test files |
| `tests/test_imports.py` | 11 | Import smoke tests for all public modules; 2 use `pytest.importorskip("ml_architectures")` |
| `tests/test_pipeline_elements.py` | 8 | `Resampler` (5), `Spectrogram` (3) |
| `tests/test_preprocessing.py` | 10 | `FilterSettings` (5), `create_spectrogram_images` (5) |
| `tests/test_training_utils.py` | 6 | `filter_unknowns` |
| `tests/test_models.py` | 12 | `Dataset_Split`, `Split` (dump/reload round-trip), `ISample`, `ITag` |
| `tests/test_usleep_prep_steps.py` | 24 | `remove_dc`, `clip_channel`, `clip_channels`, `scale_channel`, `scale_channel_manual`, `resample_channel`, `filter_channel` |
| `tests/test_metrics.py` | 19 | `kappa`, `acc`, `f1`, `get_majority_vote_predictions` |
| `tests/test_augmenters.py` | 10 | `GlobalGaussianNoise`, `RegionalGaussianNoise`, `Augmenter` |
| `tests/test_split_factories.py` | 25 | `Split.train_and_holdout`, `Split.full_test`, `Split.random`, `create_split_file` — using synthetic HDF5 files in `tmp_path` |

---

## Documentation

| File | Change |
|---|---|
| `docs/cicd.md` | **Created** — full CI/CD reference: pipeline jobs, Docker variants, pre-commit hooks, HPC deployment |
| `docs/testing.md` | **Created** — complete test suite reference: every file, every test, fixture design, skipped-test explanation |
| `docs/changelog.md` | **Created** — this file |
| `README.md` | **Updated** — hardware-specific torch install instructions, all three GitLab private package install commands, testing section with pytest commands, expanded CI/CD section |
| `.gitignore` | **Updated** — added `.venv/`, `dist/`, `.claude/`, `.ipynb_checkpoints/`; removed duplicates; replaced `.neptune/` with `mlruns/` |
