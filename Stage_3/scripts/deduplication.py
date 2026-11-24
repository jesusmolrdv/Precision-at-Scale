import argparse
from src.deduplication import Deduplication
from src.cuml_knn import CuMLSelfKnn
from src.embedding_generator import EmbeddingReaderNewWDS
from src.embedding_generator import CustomWebDataset


def main():
    parser = argparse.ArgumentParser(description="Run Deduplication Curation")
    parser.add_argument(
        "--df_paths",
        type=str,
        required=True,
        help="Paths to the dataframes (comma separated). Expected to already be a "
        "single, unified, source-tagged input (see Stage_2/scripts/combine_wds_sources.py). "
        "Deduplication runs once, on all sources combined, not once per source.",
    )
    parser.add_argument(
        "--tar_dir",
        type=str,
        required=True,
        help="Directory of tar files.",
    )
    parser.add_argument(
        "--embedding_dir",
        type=str,
        required=True,
        help="Directory of embeddings.",
    )
    parser.add_argument(
        "--output_df_path",
        type=str,
        required=True,
        help="Path to save the output dataframe",
    )
    parser.add_argument(
        "--output_adj_matrix_path",
        type=str,
        required=True,
        help="Path to save the adjacency matrix",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Cosine-similarity threshold for connected components (0.6).",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of nearest neighbors in the k-NN graph (64).",
    )
    parser.add_argument(
        "--knn_algorithm",
        type=str,
        default="ivfpq",
        choices=["brute", "ivfpq"],
        help="k-NN index. 'ivfpq' builds the approximate inverted-file/"
        "product-quantised graph; 'brute' computes exact neighbours.",
    )
    parser.add_argument(
        "--ivfpq_nlist",
        type=int,
        default=None,
        help="IVF-PQ: number of cells the dataset is partitioned into. "
        "Defaults to sqrt(N), capped at 1024, computed from the number of "
        "embeddings at fit time.",
    )
    parser.add_argument(
        "--ivfpq_nprobe",
        type=int,
        default=32,
        help="IVF-PQ: number of cells searched per query.",
    )
    parser.add_argument(
        "--ivfpq_m",
        type=int,
        default=64,
        help="IVF-PQ: number of sub-quantizers. Must divide the embedding "
        "size (1024).",
    )
    parser.add_argument(
        "--ivfpq_n_bits",
        type=int,
        default=8,
        help="IVF-PQ: bits per sub-quantizer.",
    )
    args = parser.parse_args()

    df_paths = args.df_paths.split(",")

    algo_params = None
    if args.knn_algorithm == "ivfpq":
        algo_params = {
            "nlist": args.ivfpq_nlist,
            "nprobe": args.ivfpq_nprobe,
            "M": args.ivfpq_m,
            "n_bits": args.ivfpq_n_bits,
            "usePrecomputedTables": False,
        }

    knn_method = CuMLSelfKnn(
        k=args.k,
        metric="cosine",
        algorithm=args.knn_algorithm,
        algo_params=algo_params,
    )
    embedding_source = EmbeddingReaderNewWDS(
        embedding_size=1024,
        embedding_dir=args.embedding_dir,
        dataset=CustomWebDataset(df_path=df_paths),
    )

    deduplicator = Deduplication(
        df_paths=df_paths,
        tar_dir=args.tar_dir,
        output_df_path=args.output_df_path,
        knn_method=knn_method,
        threshold=args.threshold,
        embedding_source=embedding_source,
        output_adj_matrix_path=args.output_adj_matrix_path,
    )

    deduplicator()


if __name__ == "__main__":
    main()
