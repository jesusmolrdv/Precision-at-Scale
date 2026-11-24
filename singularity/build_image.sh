#!/usr/bin/env bash
# Build one of this repo's Singularity images on a SLURM cluster.
#
# Submits the build itself as a SLURM job (not run interactively), via a
# plain `sbatch` heredoc.
#
# The build's scratch space defaults to a subdirectory under your home
# directory rather than the default /tmp, since some of these builds
# (nvcr.io/nvidia/pytorch-based, CUDA extension compilation) can use several
# GB of scratch space during the build and /tmp is often small on shared
# clusters. Override SINGULARITY_BUILD_TMPDIR_ROOT if your cluster needs
# scratch space pointed elsewhere (e.g. a larger shared filesystem).
#
# SBATCH account/partition/QoS default to placeholder values below;
# override them via SBATCH_ACCOUNT, SBATCH_PARTITION, SBATCH_QOS to match your
# own cluster.
#
# Usage (from the repo root):
#   singularity/build_image.sh <image_name>
#   SINGULARITY_BUILD_TMPDIR_ROOT=/some/other/path singularity/build_image.sh <image_name>
#
# <image_name> is one of: llama_cpp_python, clip_retrieval, torch_tensorrt, tmars

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${1:-}"
if [[ -z "${IMAGE}" ]]; then
    echo "Usage: $0 <image_name>" >&2
    echo "Valid image names: $(find "${SCRIPT_DIR}" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort | tr '\n' ' ')" >&2
    exit 1
fi

DEF_FILE="${SCRIPT_DIR}/${IMAGE}/singularity.def"
SIF_FILE="${SCRIPT_DIR}/${IMAGE}/${IMAGE}.sif"

if [[ ! -f "${DEF_FILE}" ]]; then
    echo "No singularity.def found at ${DEF_FILE}" >&2
    exit 1
fi

# Make sure this path has enough free space for a full image build (several GB).
TMPDIR_ROOT="${SINGULARITY_BUILD_TMPDIR_ROOT:-$HOME/singularity_build_tmp}"
BUILD_TMPDIR="${TMPDIR_ROOT}/${IMAGE}"

echo "Image:            ${IMAGE}"
echo "Def file:          ${DEF_FILE}"
echo "Output sif:        ${SIF_FILE}"
echo "SINGULARITY_TMPDIR: ${BUILD_TMPDIR}"
echo

mkdir -p "${BUILD_TMPDIR}"
OUTPUT_DIR="${SCRIPT_DIR}/../output"
mkdir -p "${OUTPUT_DIR}"

# --fakeroot avoids needing root for the build's %post steps; --force
# overwrites an existing .sif of the same name.
JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -A ${SBATCH_ACCOUNT:-your-account}
#SBATCH -p ${SBATCH_PARTITION:-your-partition}
#SBATCH --qos=${SBATCH_QOS:-your-qos}
#SBATCH -J "build_${IMAGE}"
#SBATCH --error=${OUTPUT_DIR}/%j.err
#SBATCH --output=${OUTPUT_DIR}/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2GB
#SBATCH -t 12:00:00

SINGULARITY_TMPDIR="${BUILD_TMPDIR}" singularity build --fakeroot --force "${SIF_FILE}" "${DEF_FILE}"
EOF
)

echo "Submitted as SLURM job ${JOB_ID}."
echo "Output: ${OUTPUT_DIR}/${JOB_ID}.out (and .err)"
echo "Once it completes, the image will be at: ${SIF_FILE}"
