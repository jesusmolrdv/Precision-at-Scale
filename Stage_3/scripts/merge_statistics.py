import argparse
import os

import polars as pl

from src.combine_stats import StatisticsOODPostProcessor, StatisticReader

# Fixed prefix for StatisticReader's get_path convention
# ("{csv_creator}_{statistic}.parquet").
CSV_CREATOR = "combined"


def main(input_df_paths, dedup_df_path, output_df_path, eval_overlap_df_path=None):
    post_processor = StatisticsOODPostProcessor(None, None)

    final_df = None
    image_name_column = StatisticReader.image_name_column
    for input_df_path in input_df_paths:
        basename = os.path.basename(input_df_path)
        prefix = f"{CSV_CREATOR}_"
        if not basename.endswith(".parquet") or not basename.startswith(prefix):
            raise ValueError(
                f"Expected filename '{basename}' to match "
                f"'{prefix}<statistic>.parquet' (StatisticReader.get_path's "
                "convention), but got a differently-shaped path instead."
            )
        statistic = basename[len(prefix) : -len(".parquet")]

        # StatisticReader.get_path joins root_statistics_path with
        # "{csv_creator}_{statistic}.parquet", so pass the input path's own
        # *directory*, not the input path itself, as root_statistics_path.
        reader = StatisticReader(CSV_CREATOR, os.path.dirname(input_df_path), statistic)
        reader.image_name_column = "key"
        image_name_column = reader.image_name_column
        df = reader.read(lazy=True)

        final_df = df if final_df is None else final_df.join(df, on=image_name_column, how="inner")

    final_df = post_processor(
        input_df=final_df.collect(), return_df=True, write_df=False
    )

    df_filtering_dedup = pl.read_parquet(dedup_df_path).filter(
        pl.col("Selected") == True
    )
    if eval_overlap_df_path is not None:
        kept = (
            pl.read_parquet(eval_overlap_df_path)
            .filter(pl.col("Selected") == True)
            .select(image_name_column)
        )
        df_filtering_dedup = df_filtering_dedup.join(
            kept, on=image_name_column, how="inner"
        )
    df_final = df_filtering_dedup.join(final_df, on=image_name_column, how="left")
    df_final.write_parquet(output_df_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_df_paths",
        type=str,
        required=True,
        help="Comma separated list of paths to the input statistics files, one per "
        "CLIPN variant (e.g. combined_detect_ood_clipn.parquet, "
        "combined_detect_ood_clipn_queries.parquet, ...).",
    )
    parser.add_argument(
        "--dedup_df_path",
        type=str,
        required=True,
        help="Path to the deduplication statistics file.",
    )
    parser.add_argument(
        "--eval_overlap_df_path",
        type=str,
        default=None,
        help="Optional path to remove_eval_overlap.py's output parquet; rows "
        "it did not select are dropped.",
    )
    parser.add_argument(
        "--output_df_path", type=str, required=True, help="Path to the output file."
    )
    args = parser.parse_args()

    input_df_paths = args.input_df_paths.split(",")
    main(
        input_df_paths,
        args.dedup_df_path,
        args.output_df_path,
        args.eval_overlap_df_path,
    )
