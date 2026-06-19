# CI/CD Pipeline

This document covers the complete continuous integration and deployment setup for common-sleep-data-pipeline.

---

## Table of contents

1. [Overview](#overview)
2. [CI pipeline](#ci-pipeline)
   - [Job: lint](#job-lint)
   - [Job: build](#job-build)
   - [Job: test](#job-test)
3. [CD pipeline](#cd-pipeline)
   - [When it triggers](#when-it-triggers)
   - [What it builds](#what-it-builds)
   - [Using the image on LUMI](#using-the-image-on-lumi)
4. [Pre-commit hooks](#pre-commit-hooks)
5. [Docker image](#docker-image)
6. [HPC deployment script](#hpc-deployment-script)
7. [Local development setup](#local-development-setup)
8. [External dependency: ml_architectures](#external-dependency-ml_architectures)
9. [Adding new jobs or steps](#adding-new-jobs-or-steps)
10. [Troubleshooting](#troubleshooting)

---

## Overview

```
Every push / pull request
         │
         ├─► lint   ruff check + ruff format --check     ~30 s
         ├─► build  python -m build --wheel               ~60 s
         └─► test   pytest, Python 3.11 + 3.12 matrix    ~8 min

git tag v*.*.*
         │
         └─► cd     Build Docker image → push to GHCR    ~10 min
```

All CI jobs run on `ubuntu-latest` GitHub-hosted runners (free tier). The test job uses **CPU-only PyTorch** to avoid downloading the 2–3 GB CUDA wheel on every run.

Jobs are independent and run in parallel. A push that fails lint does not block the test job from starting — all three results are reported together.

---

## CI pipeline

**File:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

**Triggers:**
- Every push to any branch
- Every pull request targeting `main`

Concurrent runs on the same branch are automatically cancelled (only the latest commit runs), controlled by the `concurrency` block at the top of the workflow file.

---

### Job: lint

**Purpose:** Catch code style issues and import ordering problems before they reach review.

**Tools used:**

| Tool | Role |
|------|------|
| `ruff check` | Linting — catches undefined names (F), unused imports (F401), basic style (E) |
| `ruff format --check` | Formatting — enforces consistent style (replaces black) |

**Configuration** (in `pyproject.toml`):
```toml
[tool.ruff]
line-length = 120       # wider than the 88-char black default — ML code has long lines
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]   # pycodestyle errors, pyflakes, isort
ignore = ["E501"]           # line-length enforced by formatter, not linter

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]   # re-exports in __init__ are intentional unused imports
```

---

### Job: build

**Purpose:** Verify that the package can be packaged into a wheel without errors. This catches issues like missing files, malformed `pyproject.toml`, or import cycles that prevent the package from being built.

**What it does:**
1. Installs the `build` frontend (`pip install build`) — no runtime dependencies needed.
2. Runs `python -m build --wheel`, which invokes setuptools in an isolated build environment.
3. Uploads the resulting `.whl` as a GitHub Actions artifact (retained for 7 days).

---

### Job: test

**Purpose:** Run the full test suite against two Python versions to catch version-specific regressions.

**Matrix:**
```yaml
python-version: ["3.11", "3.12"]
```

`fail-fast: false` means both matrix entries run even if one fails, so you see the full picture of what broke.

**Why CPU-only PyTorch in CI:**

The project depends on `torch~=2.11.0`. The standard PyTorch CUDA wheel is approximately 2–3 GB. On a GitHub-hosted runner this would add 10–15 minutes of download time per job.

The workaround is to install the CPU-only variant first, before `pip install -e ".[dev]"`:

```yaml
- name: Install PyTorch (CPU-only)
  run: pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu

- name: Install package + dev extras
  run: pip install -e ".[dev]"

- name: Install ml_architectures
  run: pip install git+https://github.com/RuneSchroeder/ml_architectures.git@main
```

**`ml_architectures` is installed in CI** — see [External dependency: ml_architectures](#external-dependency-ml_architectures). All 123 tests run; none are skipped.

---

## CD pipeline

**File:** [`.github/workflows/cd.yml`](../.github/workflows/cd.yml)

### When it triggers

Only on version tags pushed to the repository:

```bash
git tag v1.2.3
git push --tags
```

Tag format must match `v*.*.*` (semver). Pushes to branches never trigger the CD pipeline.

### What it builds

1. Logs in to the **GitHub Container Registry** (GHCR) at `ghcr.io` using `GITHUB_TOKEN`.
2. Extracts image tags using `docker/metadata-action`:
   - `ghcr.io/rune1750/common-sleep-data-pipeline:1.2.3`
   - `ghcr.io/rune1750/common-sleep-data-pipeline:1.2`
   - `ghcr.io/rune1750/common-sleep-data-pipeline:git-<sha>`
3. Builds the Docker image with `PYTORCH_CHANNEL=cu121` (CUDA 12.1).
4. Pushes all tags to GHCR.
5. Uses GitHub Actions layer cache so repeated builds only rebuild changed layers.

### Using the image on LUMI

LUMI uses AMD Instinct MI250X GPUs, which require **ROCm**, not CUDA. For LUMI, build locally with the ROCm channel:

```bash
docker build --build-arg PYTORCH_CHANNEL=rocm6.2 \
  -t ghcr.io/rune1750/common-sleep-data-pipeline:v1.2.3-rocm .
docker push ghcr.io/rune1750/common-sleep-data-pipeline:v1.2.3-rocm
```

Then on LUMI, convert to Singularity:

```bash
# On a LUMI login node
singularity pull csdp_v1.2.3.sif \
  docker://ghcr.io/rune1750/common-sleep-data-pipeline:v1.2.3-rocm
```

The `.sif` file can then be used in the existing SLURM scripts in `csdp_training/slurm_scripts/LUMI/`.

---

## Pre-commit hooks

**File:** [`.pre-commit-config.yaml`](../.pre-commit-config.yaml)

**One-time setup:**
```bash
pip install pre-commit
pre-commit install     # installs the hook into .git/hooks/pre-commit
```

**Hooks configured:**

| Hook | What it does |
|------|-------------|
| `ruff` (with `--fix`) | Auto-fixes import ordering and simple lint issues |
| `ruff-format` | Auto-formats the file (equivalent to running black) |
| `check-yaml` | Validates YAML files |
| `check-toml` | Validates TOML files |
| `end-of-file-fixer` | Ensures files end with a newline |
| `trailing-whitespace` | Strips trailing spaces |
| `check-merge-conflict` | Prevents committing unresolved merge conflict markers |
| `debug-statements` | Prevents committing `breakpoint()` or `pdb.set_trace()` |
| `check-added-large-files` | Blocks files over 500 KB (guards against accidentally committing data) |

---

## Docker image

**File:** [`Dockerfile`](../Dockerfile)

**Build arguments:**

| Argument | Default | Options |
|----------|---------|---------|
| `PYTORCH_CHANNEL` | `cu121` | `cu121` (CUDA 12.1), `rocm6.2` (AMD/LUMI), `cpu` (CPU-only) |

**Build examples:**
```bash
# NVIDIA CUDA (default — for PRIME cluster)
docker build -t csdp:latest .

# AMD ROCm (for LUMI supercomputer)
docker build --build-arg PYTORCH_CHANNEL=rocm6.2 -t csdp:lumi .

# CPU-only (preprocessing nodes, testing)
docker build --build-arg PYTORCH_CHANNEL=cpu -t csdp:cpu .
```

**Note on `ml_architectures`:** The Docker image does not include `ml_architectures` by default. To add it to the image, append the following to the `Dockerfile` after `pip install -e .`:

```dockerfile
RUN pip install --no-cache-dir \
    git+https://github.com/RuneSchroeder/ml_architectures.git@main
```

---

## HPC deployment script

**File:** [`scripts/deploy_hpc.sh`](../scripts/deploy_hpc.sh)

```bash
# Pull latest code only
./scripts/deploy_hpc.sh lumi.csc.fi /scratch/project_123/common-sleep-data-pipeline

# Pull latest code and immediately submit a SLURM job
./scripts/deploy_hpc.sh lumi.csc.fi /scratch/project_123/common-sleep-data-pipeline \
    csdp_training/slurm_scripts/LUMI/usleep_train_single_gpu_slurm.sh
```

---

## Local development setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/rune1750/common-sleep-data-pipeline.git
cd common-sleep-data-pipeline

# 2. Create and activate the virtual environment
python -m venv .venv
source .venv/Scripts/activate    # Windows Git Bash / WSL
# or: .venv\Scripts\activate     # Windows CMD / PowerShell

# 3. Install CPU torch first (avoids downloading the 2 GB CUDA wheel)
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu

# 4. Install the package in editable mode with dev tools
pip install -e ".[dev]"

# 5. Install ml_architectures
pip install git+https://github.com/RuneSchroeder/ml_architectures.git@main

# 6. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 7. Verify everything works
pytest
```

---

## External dependency: ml_architectures

The `ml_architectures` package (USleep, LSeqSleepNet neural network definitions) is installed from GitHub in CI before running tests:

```bash
pip install git+https://github.com/RuneSchroeder/ml_architectures.git@main
```

**Impact on CI:**
- All 123 tests run — none are skipped.
- If GitHub is temporarily unreachable, the install step will fail and the test job will be blocked. In that case, investigate connectivity rather than removing the install step.

**Local environments without `ml_architectures`:**
- `import csdp_training` still succeeds — `csdp_training/__init__.py` wraps the imports in `try/except ImportError`.
- Tests that require the package (e.g., `test_import_usleep_lightning`) use `pytest.importorskip("ml_architectures")` and are **automatically skipped** locally when the package is absent.

**Impact on Docker:**
- The Docker image installs all listed dependencies from `pyproject.toml` but does not install `ml_architectures`.
- Add the GitHub install step to the Dockerfile if the image needs to run training.

---

## Adding new jobs or steps

**Adding a new CI step:** Open `.github/workflows/ci.yml` and add a `- name: ...` step inside the relevant job.

**Adding a new CI job:** Add a new top-level key under `jobs:`. If the new job depends on another (e.g., needs the built wheel), add `needs: [build]`.

**Adding a new pre-commit hook:** Add a new entry under `repos:` in `.pre-commit-config.yaml`, then run `pre-commit install`.

---

## Troubleshooting

**`ruff check` fails on existing code:**
Run `ruff check --fix .` locally to auto-fix what can be fixed automatically.

**`torch` version conflict during `pip install -e ".[dev]"`:**
```bash
pip uninstall torch -y
pip install "torch>=2.7.0" --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

**CI test job times out:**
The most common cause is pip downloading the full CUDA torch wheel instead of CPU. Check that the "Install PyTorch (CPU-only)" step ran before "Install package + dev extras".

**Docker build fails on `pip install -e .`:**
A system library may be missing. The `Dockerfile` installs `libhdf5-dev` and `libsndfile1`. If a new dependency needs an additional system library, add it to the `apt-get install` line.

**LUMI SLURM job fails with `ModuleNotFoundError: torch`:**
The Singularity image may not contain the ROCm variant of torch. Rebuild with `--build-arg PYTORCH_CHANNEL=rocm6.2` and regenerate the `.sif` file on LUMI.
