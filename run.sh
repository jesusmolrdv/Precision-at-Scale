#!/usr/bin/env bash
# Runs a Precision-at-Scale pipeline command inside the right container image,
# so callers never need to know which of the four images a step needs.
#
# Usage:
#   ./run.sh <step> [--runtime singularity|docker] [--image-dir DIR] -- <command...>
#
# Steps and the image/conda-env each maps to:
#   stage1              -> llama_cpp_python  (Concept Generation, Stage_1/)
#   stage2-prompts      -> llama_cpp_python  (prompt generation, Stage_2/scripts/main_prompts.py)
#   stage2-web          -> clip_retrieval    (web retrieval, Stage_2/)
#   stage2-download     -> clip_retrieval    (img2dataset download of retrieved URLs)
#   stage2-web-wds      -> clip_retrieval    (web images to webdataset, Stage_2/scripts/generate_wds_from_web.py)
#   stage2-synth        -> torch_tensorrt    (synthetic generation, Stage_2/)
#   stage2-synth-wds    -> clip_retrieval    (synthetic images to webdataset, Stage_2/scripts/generate_wds_from_synth.py)
#   stage2-combine      -> clip_retrieval    (combine sources for Stage 3, Stage_2/scripts/combine_wds_sources.py)
#   stage3-embeddings   -> clip_retrieval    (SSCD embeddings, Stage_3/)
#   stage3-eval-overlap -> torch_tensorrt    (evaluation-set overlap removal, Stage_3/)
#   stage3-dedup        -> torch_tensorrt    (deduplication, Stage_3/)
#   stage3-textdetect   -> tmars             (OOD text detection, Stage_3/)
#   stage3-clipn        -> clip_retrieval    (OOD CLIPN, Stage_3/)
#   stage3-merge-stats  -> clip_retrieval    (statistics merging, Stage_3/scripts/merge_statistics.py)
#   stage3-pareto       -> clip_retrieval    (pareto filtering, Stage_3/scripts/pareto_filtering.py)
#   stage3-export       -> clip_retrieval    (final dataset export, Stage_3/scripts/export_dataset.py)
#
# Examples:
#   ./run.sh stage1 -- python main.py --config config/food.yaml --output_dir output/food --num_seeds 50
#   ./run.sh stage2-web --runtime docker -- python scripts/main_web.py --query_files ...
#
# Image files/tags are expected at:
#   Singularity: $IMAGE_DIR/<image_name>/<image_name>.sif   (default IMAGE_DIR=singularity)
#   Docker:      pas-<image_name>:latest (build once with: docker build -t pas-<image_name> singularity/<image_name>)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="singularity"
IMAGE_DIR="${IMAGE_DIR:-$SCRIPT_DIR/singularity}"
VALID_STEPS="stage1, stage2-prompts, stage2-web, stage2-download, stage2-web-wds, stage2-synth, stage2-synth-wds, stage2-combine, stage3-embeddings, stage3-eval-overlap, stage3-dedup, stage3-textdetect, stage3-clipn, stage3-merge-stats, stage3-pareto, stage3-export"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <step> [--runtime singularity|docker] [--image-dir DIR] -- <command...>" >&2
    echo "Valid steps: $VALID_STEPS" >&2
    exit 1
fi

STEP="$1"; shift

while [[ $# -gt 0 && "$1" != "--" ]]; do
    case "$1" in
        --runtime) RUNTIME="$2"; shift 2 ;;
        --image-dir) IMAGE_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
[[ "${1:-}" == "--" ]] && shift
CMD=("$@")

if [[ ${#CMD[@]} -eq 0 ]]; then
    echo "No command given after '--'." >&2
    exit 1
fi

case "$STEP" in
    stage1)             IMAGE="llama_cpp_python"; CONDA_ENV="" ;;
    stage2-prompts)      IMAGE="llama_cpp_python"; CONDA_ENV="" ;;
    stage2-web)          IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage2-download)      IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage2-web-wds)         IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage2-synth)             IMAGE="torch_tensorrt";   CONDA_ENV="" ;;
    stage2-synth-wds)          IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage2-combine)              IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage3-embeddings)      IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage3-eval-overlap)      IMAGE="torch_tensorrt";   CONDA_ENV="" ;;
    stage3-dedup)             IMAGE="torch_tensorrt";   CONDA_ENV="" ;;
    stage3-textdetect)          IMAGE="tmars";            CONDA_ENV="tmars" ;;
    stage3-clipn)                 IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage3-merge-stats)             IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage3-pareto)                 IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    stage3-export)                 IMAGE="clip_retrieval";   CONDA_ENV="clipretrieval" ;;
    *)
        echo "Unknown step '$STEP'." >&2
        echo "Valid steps: $VALID_STEPS" >&2
        exit 1
        ;;
esac

# PYTHONPATH=. is required: Stage_2/Stage_3 scripts do `from src.X import Y`
# where `src/` is a sibling of `scripts/`, not a subdirectory of it, so
# `python scripts/foo.py` alone can't find it (Python puts the invoked
# script's own directory on sys.path, not the caller's cwd). Harmless to set
# for steps that don't need it.
#
# PYTHONNOUSERSITE=1 is required too: Singularity binds $HOME by default, and
# without this, Python's user-site mechanism picks up packages from the
# host's ~/.local/lib/pythonX.Y/site-packages ahead of the container's own
# site-packages, e.g. an older host-level `protobuf` shadowing the one
# `onnx` needs inside torch_tensorrt, otherwise it can fail with
# "cannot import name 'builder' from 'google.protobuf.internal'".
#
# torch_tensorrt has no conda at all. It's the nvcr.io base pytorch image's
# system python3/pip environment, not a conda install, so there's no
# /opt/conda/bin/activate to source. llama_cpp_python is the same (system
# python3). Everything else activates its named env.
# HF_HOME is required too: if it resolves to a path outside the bound
# project tree (e.g. via a symlinked $HOME/.cache), huggingface_hub's
# download code fails with FileNotFoundError trying to create it inside
# the container's mount namespace. Point it at a cache dir that's always
# inside what we actually bind below.
#
# TORCH_HOME is the same fix for torch.hub's own checkpoint cache
# (~/.cache/torch by default), needed by stage3-textdetect's mmocr model
# checkpoint download. MPLCONFIGDIR is the same fix again for matplotlib,
# which stage3-textdetect pulls in indirectly and which warns on startup
# when it can't create its own ~/.cache/matplotlib.
#
# --cleanenv (below) also strips any other host env var, including ones a
# caller sets deliberately, e.g. SD21_BASE_LOCAL_PATH (Stage_2/README.md's
# workaround for the stable-diffusion-2-1-base HF outage). Forward it
# through explicitly if set, same as HF_HOME/TORCH_HOME.
BIND_ROOT="${PROJECT_ROOT:-$(pwd)}"
printf -v CMD_QUOTED '%q ' "${CMD[@]}"

EXTRA_ENV=""
if [[ -n "${SD21_BASE_LOCAL_PATH:-}" ]]; then
    printf -v EXTRA_ENV '%sSD21_BASE_LOCAL_PATH=%q ' "$EXTRA_ENV" "$SD21_BASE_LOCAL_PATH"
fi

if [[ -n "$CONDA_ENV" ]]; then
    INNER_CMD="source activate /opt/conda/envs/$CONDA_ENV && PYTHONPATH=. PYTHONNOUSERSITE=1 HF_HOME=${BIND_ROOT}/.cache/huggingface TORCH_HOME=${BIND_ROOT}/.cache/torch MPLCONFIGDIR=${BIND_ROOT}/.cache/matplotlib ${EXTRA_ENV}${CMD_QUOTED}"
else
    INNER_CMD="PYTHONPATH=. PYTHONNOUSERSITE=1 HF_HOME=${BIND_ROOT}/.cache/huggingface TORCH_HOME=${BIND_ROOT}/.cache/torch MPLCONFIGDIR=${BIND_ROOT}/.cache/matplotlib ${EXTRA_ENV}${CMD_QUOTED}"
fi

case "$RUNTIME" in
    singularity)
        SIF="$IMAGE_DIR/$IMAGE/$IMAGE.sif"
        if [[ ! -f "$SIF" ]]; then
            echo "Singularity image not found: $SIF" >&2
            echo "(build it: singularity build $SIF $IMAGE_DIR/$IMAGE/singularity.def. On a SLURM cluster, prefer singularity/build_image.sh $IMAGE, which submits the build via SLURM.)" >&2
            exit 1
        fi
        # Bind the whole project tree, not just the cwd: pipeline steps cd
        # into e.g. Stage_2/ and then reference sibling stages via relative
        # ../Stage_1/... paths, which don't resolve if only the leaf
        # directory is bound into the container's namespace.
        #
        # --cleanenv is required too: without it, a host BASH_ENV (e.g.
        # Lmod/OpenHPC clusters) can leak into the container and shadow its
        # python3 with the host's own conda. Doesn't affect INNER_CMD's own
        # vars or --nv's GPU visibility.
        exec singularity exec --cleanenv --nv -B "$BIND_ROOT" --pwd "$(pwd)" "$SIF" bash -c "$INNER_CMD"
        ;;
    docker)
        TAG="pas-$IMAGE:latest"
        # --entrypoint bash bypasses each image's own ENTRYPOINT so INNER_CMD
        # runs via a plain `bash -c`, matching the Singularity branch.
        exec docker run --rm --gpus all --entrypoint bash \
            -v "$BIND_ROOT":"$BIND_ROOT" -w "$(pwd)" "$TAG" -c "$INNER_CMD"
        ;;
    *)
        echo "Unknown runtime '$RUNTIME' (expected singularity or docker)." >&2
        exit 1
        ;;
esac
