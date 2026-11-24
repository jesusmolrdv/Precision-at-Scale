"""Write the final PaS dataset: the images Pareto filtering selected.

Reads the Pareto output parquet, keeps the rows whose filtering column is True,
and copies those images out of the combined Stage 2 webdataset into a new
webdataset: tar shards of `<key>.jpg` + `<key>.json`, each with a parquet index
of the same name (one row per image: key, source, concept, and the prompt or
url it came from).
"""

import argparse
import glob
import os
import tarfile

import polars as pl
from tqdm import tqdm
from webdataset import TarWriter

METADATA_COLUMNS = ["key", "source", "concept", "prompt", "url", "query"]


def load_selected_keys(pareto_df_path, filtering_column):
    df = pl.read_parquet(pareto_df_path)
    if filtering_column not in df.columns:
        raise ValueError(f"{pareto_df_path} has no '{filtering_column}' column")
    keys = df.filter(pl.col(filtering_column) == True).get_column("key")
    return set(keys.cast(pl.Utf8).to_list())


def iter_selected_images(tar_path, wanted):
    """Yield (key, jpg bytes) for every image in tar_path whose key is wanted."""
    with tarfile.open(tar_path) as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            key, _, ext = name.partition(".")
            if ext == "jpg" and key in wanted:
                yield key, tar.extractfile(member).read()


class ShardWriter:
    def __init__(self, output_dir, shard_size):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard = -1
        self.tar = None
        self.rows = []

    def _close(self):
        if self.tar is not None:
            self.tar.close()
            pl.DataFrame(self.rows).write_parquet(
                os.path.join(self.output_dir, f"{self.shard:06d}.parquet")
            )

    def write(self, jpg, metadata):
        if self.tar is None or len(self.rows) == self.shard_size:
            self._close()
            self.shard += 1
            self.rows = []
            self.tar = TarWriter(
                os.path.join(self.output_dir, f"{self.shard:06d}.tar")
            )
        self.tar.write({"__key__": metadata["key"], "jpg": jpg, "json": metadata})
        self.rows.append(metadata)

    def close(self):
        self._close()
        self.tar = None


def main(args):
    selected = load_selected_keys(args.pareto_df_path, args.filtering_column)
    print(f"{len(selected)} selected images in {args.pareto_df_path}")

    parquet_paths = sorted(glob.glob(os.path.join(args.tar_dir, "*.parquet")))
    if not parquet_paths:
        raise FileNotFoundError(f"No *.parquet shard indexes in {args.tar_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    writer = ShardWriter(args.output_dir, args.shard_size)
    selected_df = pl.DataFrame({"key": pl.Series(sorted(selected), dtype=pl.Utf8)})
    written = set()
    pbar = tqdm(total=len(selected))

    for parquet_path in parquet_paths:
        meta = pl.read_parquet(parquet_path)
        meta = meta.with_columns(pl.col("key").cast(pl.Utf8)).join(
            selected_df, on="key", how="semi"
        )
        if meta.is_empty():
            continue
        columns = [c for c in METADATA_COLUMNS if c in meta.columns]
        meta_by_key = {row["key"]: row for row in meta.select(columns).iter_rows(named=True)}

        tar_path = parquet_path[: -len(".parquet")] + ".tar"
        for key, jpg in iter_selected_images(tar_path, meta_by_key.keys()):
            if key in written:
                continue
            metadata = {c: meta_by_key[key].get(c) for c in METADATA_COLUMNS}
            writer.write(jpg, metadata)
            written.add(key)
            pbar.update(1)

    writer.close()
    pbar.close()

    missing = selected - written
    if missing:
        raise RuntimeError(
            f"{len(missing)} selected images were not found in {args.tar_dir}, "
            f"e.g. {sorted(missing)[:5]}"
        )
    print(f"Wrote {len(written)} images in {writer.shard + 1} shard(s) to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the images selected by Pareto filtering as a webdataset."
    )
    parser.add_argument(
        "--pareto_df_path",
        type=str,
        required=True,
        help="Pareto filtering output parquet (pareto_filtering.py --output_df_path).",
    )
    parser.add_argument(
        "--tar_dir",
        type=str,
        required=True,
        help="Combined Stage 2 webdataset directory (combine_wds_sources.py --output_dir).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for the final dataset's tar shards and parquet indexes.",
    )
    parser.add_argument(
        "--filtering_column",
        type=str,
        default="Selected",
        help="Boolean column marking the selected rows (pareto_filtering.py --output_filtering_column).",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10_000,
        help="Images per output shard.",
    )
    main(parser.parse_args())
