import argparse
import os

import polars as pl


def parse_sources(sources_arg: str) -> dict:
    sources = {}
    for item in sources_arg.split(","):
        name, path = item.split("=", 1)
        sources[name] = path
    return sources


def combine_wds_sources(sources: dict, output_dir: str) -> None:
    """
    Combine N sources' per-chunk webdataset dataframes into one source-tagged
    dataframe, without physically duplicating tar binaries.

    Each source's chunks are kept as separate, source-prefixed parquet/tar
    pairs rather than merged into a single file: downstream loaders
    (Deduplication, CLIPNCuratorNewWDS, embedding generation) already glob
    --df_paths and pl.concat(how="diagonal") the results, so a directory of
    consistently-named per-chunk files reads back as one unified,
    source-tagged dataframe without any code changes there.

    Tars are symlinked (not copied) since they can be large; parquets are
    read and rewritten (not symlinked) with a "source" column added and
    renamed to match their symlinked tar's basename, since the downstream
    loaders (CustomWebDataset, absent an explicit tar_dir) resolve tar_path
    by replacing ".parquet" with ".tar" in the parquet's own full path;
    i.e. the tar must sit next to the parquet, in output_dir itself, not a
    nested subdirectory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for source_name, source_dir in sources.items():
        parquet_names = sorted(
            p for p in os.listdir(source_dir) if p.endswith(".parquet")
        )
        for parquet_name in parquet_names:
            basename = parquet_name[: -len(".parquet")]
            src_tar_path = os.path.join(source_dir, f"{basename}.tar")
            if not os.path.exists(src_tar_path):
                raise FileNotFoundError(
                    f"Expected matching tar for {parquet_name}: {src_tar_path}"
                )

            # Source-prefixed name avoids basename collisions across sources
            # (both web/synth chunks can independently produce e.g.
            # 000000.tar).
            combined_basename = f"{source_name}__{basename}"
            dest_tar_path = os.path.join(output_dir, f"{combined_basename}.tar")
            if os.path.lexists(dest_tar_path):
                os.remove(dest_tar_path)
            os.symlink(os.path.abspath(src_tar_path), dest_tar_path)

            df = pl.read_parquet(os.path.join(source_dir, parquet_name))
            df = df.with_columns(pl.lit(source_name).alias("source"))
            dest_parquet_path = os.path.join(
                output_dir, f"{combined_basename}.parquet"
            )
            df.write_parquet(dest_parquet_path)

    print(
        f"Combined {len(sources)} source(s) ({', '.join(sources)}) into "
        f"{output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-source Stage 2 webdataset outputs (e.g. web + "
        "synthetic) into one source-tagged dataframe/tar-dir pair for Stage 3."
    )
    parser.add_argument(
        "--sources",
        type=str,
        required=True,
        help="Comma-separated name=dir pairs, e.g. "
        "web=output/food/images/web_wds,synthetic=output/food/images/synth_wds. "
        "N-way, not limited to 2.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory. Source-prefixed parquet files are written, and "
        "matching tar files symlinked (not copied), directly here.",
    )
    args = parser.parse_args()

    sources = parse_sources(args.sources)
    os.makedirs(args.output_dir, exist_ok=True)
    combine_wds_sources(sources, args.output_dir)


if __name__ == "__main__":
    main()
