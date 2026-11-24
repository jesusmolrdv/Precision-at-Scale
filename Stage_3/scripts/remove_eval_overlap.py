import argparse
import glob
import os

import numpy as np
import polars as pl
import torch

from src.cuml_knn import CUMLKnn


def load_chunks(embedding_dir, with_metadata=True):
    """Load the chunk_*.pt/.parquet pairs written by generate_sscd_embeddings.py.

    Chunks are concatenated in ascending chunk order, the same order
    EmbeddingReaderNewWDS uses, so embeddings and metadata rows line up.
    """
    pt_paths = sorted(glob.glob(os.path.join(embedding_dir, "chunk_*.pt")))
    if not pt_paths:
        raise FileNotFoundError(f"No chunk_*.pt files found in {embedding_dir}")

    embeddings = np.concatenate([torch.load(p).numpy() for p in pt_paths])

    metadata = None
    if with_metadata:
        parquet_paths = [p[: -len(".pt")] + ".parquet" for p in pt_paths]
        missing = [p for p in parquet_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing embedding metadata: {missing[0]}")
        metadata = pl.concat(
            [pl.read_parquet(p) for p in parquet_paths], how="diagonal"
        )

    return metadata, embeddings


def main(args):
    metadata, embeddings = load_chunks(args.embedding_dir)
    _, reference_embeddings = load_chunks(
        args.reference_embedding_dir, with_metadata=False
    )
    print(
        f"{len(embeddings)} dataset images vs "
        f"{len(reference_embeddings)} evaluation images"
    )

    knn = CUMLKnn(reference_embeddings, k=1, metric="cosine")
    distances, _ = knn(embeddings)
    # cuML returns cosine distances; convert to similarity, the same scale as
    # deduplication.py's --threshold. An image whose similarity to its
    # nearest evaluation image exceeds the threshold is removed.
    similarity = 1.0 - np.asarray(distances)[:, 0]
    selected = similarity <= args.threshold
    print(f"Keeping {int(selected.sum())}/{len(selected)} images")

    df = metadata.select("key").with_columns(pl.Series("Selected", selected))
    df.write_parquet(args.output_df_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove images that are near-duplicates of the evaluation "
        "test sets."
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory of the dataset's SSCD embeddings.",
    )
    parser.add_argument(
        "--reference_embedding_dir",
        type=str,
        required=True,
        help="Directory of the evaluation test sets' SSCD embeddings.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.45,
        help="Cosine-similarity threshold (0.45): images more similar than "
        "this to any evaluation image are removed.",
    )
    parser.add_argument(
        "--output_df_path",
        type=str,
        required=True,
        help="Path of the output parquet (key, Selected).",
    )
    main(parser.parse_args())
