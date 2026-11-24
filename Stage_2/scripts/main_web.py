"""
Script for querying CLIP retrieval with command line arguments.

This script allows you to retrieve images using CLIP embeddings based on input queries.
It loads queries from JSON or plain text files, processes them with the specified prefix/suffix,
queries the CLIP index, and saves the filtered results to a parquet file.

Usage:
    python scripts/main_web.py --query_files <query_file1> <query_file2> ... --output_file <output.parquet>
                     [--prefix <prefix>] [--suffix <suffix>] 
                     [--indices_json_path <path>] [--columns <column1> <column2> ...]
                     [--num_images <num>] [--num_result_ids <num>]

Arguments:
    --query_files: List of paths to query files (JSON or plain text)
    --output_file: Path to the output parquet file
    --prefix: Prefix to add before each query (default: "")
    --suffix: Suffix to add after each query (default: "")
    --indices_json_path: Path to the indices.json file (default: "indices.json")
    --columns: List of columns to return in the output (default: ["url", "caption"]).
        The retrieval similarity is always added as an extra "similarity" column.
    --min_similarity: Keep only results whose similarity exceeds this value
        (the paper's lambda_Web). Default: None, i.e. no filtering
    --num_images: Number of images to retrieve per query (default: 500)
    --num_result_ids: Number of result IDs to retrieve per query (default: 500)
    --save_every: Write the output parquet every N newly-processed queries (default: 20)
    --first: Index of the first query in the list to process, for resuming/splitting a run (default: 0)
    --last: Index of the last query in the list to process (default: 1000000) -- queries beyond
        this index are silently skipped; raise it if --query_files has more than 1000000 entries

Example usage:

python scripts/main_web.py --query_files ../Stage_1/output/birds/valid_concepts.json --output_file output/birds/web_retrieval.parquet --prefix "a photo of a " --suffix "" --indices_json_path laion5b/indices.json --num_images 500 --num_result_ids 500
"""

import argparse
import os
import json
import polars as pl
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions
from tqdm import tqdm


def is_json_file(filename):
    """Check if a file is in JSON format."""
    try:
        with open(filename) as file:
            json.load(file)
        return True
    except ValueError:
        return False


def load_queries_from_file(filename):
    """Load queries from a JSON or plain text file."""
    if is_json_file(filename):
        with open(filename) as file:
            return json.load(file)
    else:
        with open(filename) as file:
            return [line.rstrip() for line in file]


def main(args):
    # clip_model/enable_hdf5/use_arrow/enable_mclip_option below are only
    # defaults: load_clip_indices() overrides any of these that --indices_json_path's
    # JSON also defines (currently all four, for laion5b/indices.json).
    clip_options = ClipOptions(
        indice_folder="",  # overridden by --indices_json_path; unused by knn.query()
        clip_model="open_clip:ViT-H-14",
        enable_hdf5=False,
        enable_faiss_memory_mapping=True,
        columns_to_return=args.columns,
        reorder_metadata_by_ivf_index=False,
        enable_mclip_option=False,
        use_jit=True,
        use_arrow=True,
        provide_safety_model=True,
        provide_violence_detector=True,
        provide_aesthetic_embeddings=False,
    )

    resources = load_clip_indices(args.indices_json_path, clip_options)
    knn_service = KnnService(clip_resources=resources)

    queries = []
    for query_file in args.query_files:
        queries.extend(load_queries_from_file(query_file))

    dfs = []
    if os.path.exists(args.output_file):
        df = pl.read_parquet(args.output_file)
        queries_run = set(df.select(["query"]).get_columns()[0].to_list())
        dfs.append(df)
    else:
        queries_run = set()

    num_new = 0
    for idx, query in enumerate(tqdm(queries)):
        if idx < args.first or idx > args.last:
            continue

        concept = query.strip()
        query = args.prefix + concept + args.suffix

        if query in queries_run:
            print(f"Skipping query {query} as it has already been run")
            continue

        results = knn_service.query(
            text_input=query,
            num_images=args.num_images,
            num_result_ids=args.num_result_ids,
        )
        results = [r for r in results if all(c in r for c in args.columns)]

        # The paper's lambda_Web: keep only results whose similarity to the
        # query embedding exceeds the threshold. Off unless --min_similarity
        # is given.
        if args.min_similarity is not None:
            results = [
                r
                for r in results
                if r.get("similarity") is not None
                and r["similarity"] > args.min_similarity
            ]

        output_columns = args.columns + ["similarity"]
        rows = [
            {**{c: e[c] for c in args.columns}, "similarity": e.get("similarity")}
            for e in results
        ]
        df = pl.from_dicts(rows, schema=output_columns)
        df = df.with_columns(
            pl.lit(query).alias("query"), pl.lit(concept).alias("concept")
        )
        dfs.append(df)
        num_new += 1

        if num_new % args.save_every == 0:
            final_df = pl.concat(dfs)
            final_df.write_parquet(args.output_file)
            del final_df

    if dfs:
        final_df = pl.concat(dfs)
        final_df.write_parquet(args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query clip retrieval with command line arguments."
    )
    parser.add_argument(
        "--query_files", nargs="+", help="Path to query files", required=True
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file name", required=True
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix to add before each query"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix to add after each query"
    )
    parser.add_argument(
        "--indices_json_path",
        type=str,
        default="indices.json",
        help="Path to indices.json",
    )
    parser.add_argument(
        "--columns", nargs="+", default=["url", "caption"], help="Columns to return"
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=None,
        help="Keep only results whose cosine similarity to the query exceeds "
        "this value (the paper's lambda_Web). Default: no filtering. The "
        "similarity is written to the output parquet either way.",
    )
    parser.add_argument(
        "--save_every", type=int, default=20, help="Save every n queries"
    )
    parser.add_argument(
        "--first", type=int, default=0, help="Index of first query to process"
    )
    parser.add_argument(
        "--last", type=int, default=1000000, help="Index of last query to process"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Number of images to retrieve per query",
    )
    parser.add_argument(
        "--num_result_ids",
        type=int,
        default=500,
        help="Number of result IDs to retrieve per query",
    )
    args = parser.parse_args()

    main(args)
