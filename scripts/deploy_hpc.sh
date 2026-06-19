#!/usr/bin/env bash
# deploy_hpc.sh — Pull latest code on an HPC login node (LUMI / PRIME / any SSH target).
#
# Usage:
#   ./scripts/deploy_hpc.sh <ssh_host> <remote_project_dir> [slurm_script]
#
# Examples:
#   ./scripts/deploy_hpc.sh lumi.csc.fi /scratch/project_123/common-sleep-data-pipeline
#   ./scripts/deploy_hpc.sh lumi.csc.fi /scratch/project_123/common-sleep-data-pipeline \
#       csdp_training/slurm_scripts/LUMI/usleep_train_single_gpu_slurm.sh
#
# Requirements:
#   - SSH key-based auth configured for <ssh_host>
#   - git is available on the remote host
#   - The remote directory is already a git checkout of this repo

set -euo pipefail

SSH_HOST="${1:?ERROR: SSH host required. Usage: $0 <ssh_host> <remote_dir> [slurm_script]}"
REMOTE_DIR="${2:?ERROR: Remote directory required.}"
SLURM_SCRIPT="${3:-}"

echo "==> Deploying to ${SSH_HOST}:${REMOTE_DIR}"

ssh "${SSH_HOST}" bash << REMOTE_COMMANDS
set -e

cd "${REMOTE_DIR}"

BRANCH=\$(git branch --show-current)
echo "  Remote branch : \${BRANCH}"
echo "  Current HEAD  : \$(git log -1 --oneline)"

git fetch origin
BEHIND=\$(git rev-list HEAD..origin/\${BRANCH} --count)
echo "  Commits behind origin: \${BEHIND}"

if [ "\${BEHIND}" -eq 0 ]; then
    echo "  Already up-to-date. Nothing to deploy."
else
    git pull --ff-only
    echo "  Updated to: \$(git log -1 --oneline)"
fi

if [ -n "${SLURM_SCRIPT}" ]; then
    echo ""
    echo "==> Submitting SLURM job: ${SLURM_SCRIPT}"
    JOB_ID=\$(sbatch "${SLURM_SCRIPT}" | awk '{print \$NF}')
    echo "  Job submitted: \${JOB_ID}"
    echo "  Monitor with: squeue -j \${JOB_ID}"
fi
REMOTE_COMMANDS

echo "==> Done."
