# Used by: Stage_2 synthetic image generation (TensorRT diffusion), Stage_3
# deduplication (embeddings + adjacency-matrix dedup).
#
# cuml/cudf for dedup come from this base image, not from anything below.
FROM nvcr.io/nvidia/pytorch:23.07-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /workdir
COPY requirements.txt /workdir/requirements.txt

RUN python3 -m pip install --upgrade pip \
    # --- Stage_2 synthetic generation dependencies ---
    && python3 -m pip install --upgrade tensorrt==8.6.1 \
    && python3 -m pip install webdataset \
    && pip3 install -r /workdir/requirements.txt \
    # --- Stage_3 deduplication dependencies ---
    && pip install "numpy<2" faiss-cpu polars \
    && apt update --allow-unauthenticated --allow-insecure-repositories \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt update --allow-unauthenticated --allow-insecure-repositories \
    && apt install -y g++-11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash", "-c", "source activate /opt/conda && \"$@\"", "--"]
