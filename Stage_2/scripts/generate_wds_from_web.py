import os
import argparse
import glob

import polars as pl
from PIL import Image
from tqdm import tqdm
from braceexpand import braceexpand
import webdataset as wds
from webdataset import TarWriter


def prepare_dataframes(input_path, queries_path):
    # Load queries data. The same url can match multiple queries -- keep only
    # the first match per url so the join below doesn't duplicate images.
    df_queries = pl.read_parquet(list(braceexpand(queries_path))).unique(
        subset=["url"], keep="first"
    )
    print(f"Loaded {len(df_queries)} query records with columns: {df_queries.columns}")

    # Load and concatenate parquet files containing result data
    parquet_files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    df_results = (
        pl.concat(
            [
                pl.read_parquet(pf)
                .filter(pl.col("status") == "success")
                .with_columns(pl.lit(pf).alias("path"))
                for pf in parquet_files
            ]
        )
        .with_columns(
            pl.col("path")
            .str.replace(input_path, "", literal=True)
            .str.replace(".parquet", "", literal=True)
            .str.replace("/", "", literal=True)
            .alias("download_chunk")
        )
        .drop("path")
    )
    print(f"Loaded {len(df_results)} result records with columns: {df_results.columns}")

    # Merge result data with query data
    df = df_results.join(df_queries, on="url", how="inner").drop(
        "status",
        "error_message",
        "width",
        "height",
        "original_width",
        "original_height",
        "exif",
        "sha256",
        strict=False,
    )
    print(f"Merged dataframe has {len(df)} records with columns: {df.columns}")

    # Add new_img_index and chunk_id columns
    df = df.with_row_index("new_img_index")
    df = df.with_columns((pl.col("new_img_index") // 100_000).alias("chunk_id"))

    # Convert all columns to strings and pad new_img_index with zeros
    for col in df.columns:
        df = df.with_columns(pl.col(col).cast(str))
    df = df.with_columns(pl.col("new_img_index").str.zfill(9))

    print(f"Final dataframe has {len(df)} records with columns: {df.columns}")
    return df


def write_tar_and_parquet(df, input_path, output_path):
    # Initialize tracking variables
    last_chunk = -1
    pbar = tqdm(total=len(df))

    for row in df.iter_rows(named=True):
        chunk_id = int(row["chunk_id"])
        if chunk_id != last_chunk:
            if last_chunk != -1:
                writer.close()
                pl.DataFrame(new_rows).write_parquet(
                    f"{output_path}/{last_chunk:06d}.parquet"
                )
            writer = TarWriter(f"{output_path}/{chunk_id:06d}.tar")
            new_rows = []
            last_chunk = chunk_id

        # Prepare the dictionary to write to the TAR file
        data_dict = {
            "key": row["new_img_index"],
            "n": chunk_id,
            "ordinal_key": row["new_img_index"],
            "old_download_chunk": row["download_chunk"],
            "old_key": row["key"],
            "url": row["url"],
            "query": row.get("query"),
            # Retrieval outputs from before the concept column was added
            # only have the query (prefix + concept + suffix).
            "concept": row.get("concept", row.get("query")),
        }
        new_rows.append(data_dict)

        # Build image path and check if image exists
        img_path = os.path.join(input_path, row["download_chunk"], f"{row['key']}.jpg")
        assert os.path.exists(img_path), f"Image not found: {img_path}"

        # Load and write the image to the TAR file
        img = Image.open(img_path)
        writer.write({"__key__": row["new_img_index"], "jpg": img, "json": data_dict})
        pbar.update(1)

    # Close the last TAR file and save the final parquet
    if last_chunk != -1:
        writer.close()
        pl.DataFrame(new_rows).write_parquet(f"{output_path}/{last_chunk:06d}.parquet")


def main(args):
    # Prepare dataframes and write TAR and parquet files
    df = prepare_dataframes(args.input, args.queries)
    write_tar_and_parquet(df, args.input, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare WebDataset TARs from Parquet files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input directory containing parquet files.",
    )
    parser.add_argument(
        "--queries", type=str, required=True, help="Path to the queries parquet file."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output directory."
    )
    args = parser.parse_args()
    main(args)
