FROM condaforge/miniforge3:23.11.0-0

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y build-essential git curl ca-certificates wget pkg-config libturbojpeg0-dev libopencv-dev ffmpeg libsm6 libxext6 libgeos-dev \
    && apt-get clean

# If CUDA 12.1 doesn't match the host driver, use pytorch-cuda=11.8 (and add -c conda-forge) instead.
RUN mamba create -y --name clipretrieval python=3.9 pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=12.1 -c pytorch -c nvidia \
    && bash -c "source activate /opt/conda/envs/clipretrieval && pip install clip-retrieval==2.44.0 autokeras==1.0.18 keras==2.8.0 Keras-Preprocessing==1.1.2 tensorflow==2.8.0" \
    # clip-retrieval pulls in transformers unpinned; a current transformers calls
    # torch.utils._pytree.register_pytree_node, which doesn't exist until torch
    # 2.2 (we're pinned to 2.1.2 above), so pin to a version that still uses the
    # older _register_pytree_node private API.
    && bash -c "source activate /opt/conda/envs/clipretrieval && pip install 'transformers==4.30.2'" \
    # clip-retrieval also pulls in sentence-transformers unpinned; a current
    # one imports transformers.utils.import_utils.is_nltk_available, which
    # doesn't exist in the transformers version pinned above, so pin to a
    # version contemporaneous with transformers==4.30.2.
    && bash -c "source activate /opt/conda/envs/clipretrieval && pip install 'sentence-transformers==2.2.2'" \
    # sentence-transformers==2.2.2 itself imports cached_download from
    # huggingface_hub, which is pulled in unpinned and was on a version
    # where that's already removed (gone since 0.26.0), so pin to a version
    # contemporaneous with the rest of this pin chain that still has it.
    && bash -c "source activate /opt/conda/envs/clipretrieval && pip install 'huggingface_hub==0.16.4'" \
    # tensorflow==2.8.0's bundled *_pb2.py files were generated against an old
    # protoc; a current protobuf (>=3.20, using the new C++-backed
    # implementation) refuses to load them at runtime with "Descriptors
    # cannot be created directly." Pin to the last 3.19.x release that
    # clip_back.py's load_safety_model() (autokeras/tensorflow import chain)
    # needs in order to actually import tensorflow.
    && bash -c "source activate /opt/conda/envs/clipretrieval && pip install 'protobuf==3.19.6'" \
    && bash -c "source activate /opt/conda/envs/clipretrieval && pip uninstall -y faiss faiss-cpu" \
    # albumentations>=1.4.23 dropped the top-level A.longest_max_size that
    # img2dataset 1.45.0's resizer.py still calls directly (AttributeError);
    # pin to the last known-good release.
    && bash -c "source activate /opt/conda/envs/clipretrieval && pip install polars pandas tqdm webdataset paretoset kneed 'albumentations==1.4.22' img2dataset braceexpand funcy loguru joblib" \
    && bash -c "source activate /opt/conda/envs/clipretrieval && mamba install -y -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl" \
    && bash -c "source activate /opt/conda/envs/clipretrieval && conda clean --all -y && pip cache purge"

ENTRYPOINT ["/bin/bash", "-c", "source activate /opt/conda/envs/clipretrieval && \"$@\"", "--"]
