import os
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, save_npz, load_npz
from typing import Union, List
from .utils import select_first_in_group, connected_components


def _build_adj_matrix(
    knn_method,
    embeddings,
    is_similarity: bool = False,
    output_adj_matrix_path: str = None,
):
    distances, knn = knn_method(embeddings)

    if not is_similarity:
        distances = 1 - distances  # Convert distances to similarities

    size = len(knn)
    adj = csr_matrix((size, size), dtype=float)

    i_indices = np.repeat(np.arange(size), len(knn[0]))
    j_indices = np.concatenate(knn)
    adj[i_indices, j_indices] = np.concatenate(distances)
    adj = adj.maximum(adj.transpose())

    if output_adj_matrix_path is not None:
        save_npz(output_adj_matrix_path, adj)

    return adj


def _find_ccs(adj, threshold):
    bin_adj = adj.copy()
    bin_adj.data = bin_adj.data > threshold
    bin_adj.eliminate_zeros()
    bin_adj = bin_adj.astype(bool)

    n_components, labels = connected_components(bin_adj)
    return n_components, labels


class Deduplication:
    def __init__(
        self,
        df_paths: Union[str, List[str]],
        tar_dir: str,
        output_df_path: str,
        knn_method,
        threshold: float,
        embedding_source,
        output_adj_matrix_path: str = None,
        output_filtering_column: str = "Selected",
        df_key_column: str = "key",
        remove_columns: List[str] = [],
        is_similarity: bool = False,
    ) -> None:
        if isinstance(df_paths, str):
            self.df_paths = [df_paths]
        else:
            self.df_paths = df_paths

        self.df_paths = sorted(self.df_paths)
        self.tar_dir = tar_dir
        self.output_df_path = output_df_path
        self.output_filtering_column = output_filtering_column
        self.knn_method = knn_method
        self.threshold = threshold
        self.embedding_source = embedding_source
        self.output_adj_matrix_path = output_adj_matrix_path
        self.is_similarity = is_similarity
        self.df_key_column = df_key_column

        def load_df(df_path, tar_dir):
            if df_path.endswith(".csv"):
                return pl.read_csv(df_path).with_columns(
                    pl.lit(
                        os.path.join(
                            tar_dir,
                            os.path.basename(df_path).replace(".csv", ".tar"),
                        )
                    ).alias("tar_path")
                )
            elif df_path.endswith(".parquet"):
                return pl.read_parquet(df_path).with_columns(
                    pl.lit(
                        os.path.join(
                            tar_dir,
                            os.path.basename(df_path).replace(".parquet", ".tar"),
                        )
                    ).alias("tar_path")
                )

        # "diagonal" since a unified input can combine chunks from sources with
        # disjoint schemas (e.g. web vs synthetic) — see
        # Stage_2/scripts/combine_wds_sources.py.
        self.df = pl.concat(
            [load_df(df_path, tar_dir) for df_path in self.df_paths], how="diagonal"
        )
        self.df = self.df.drop(remove_columns)
        if isinstance(self.df, pl.LazyFrame):
            self.df = self.df.collect()

    def __call__(self, *args, **kwargs):
        embeddings = self.embedding_source.get_filtered_embeddings(
            filtering_df=self.df, format="numpy"
        )

        # A cached adjacency matrix is only reusable if it was built with the
        # same k *and* the same k-NN index, so the sidecar records both.
        k = getattr(self.knn_method, "k", None)
        algorithm = getattr(self.knn_method, "algorithm", None)
        signature = None if k is None else f"{k}|{algorithm}"
        k_sidecar_path = (
            self.output_adj_matrix_path + ".k"
            if self.output_adj_matrix_path is not None
            else None
        )
        cached_signature = None
        if k_sidecar_path is not None and os.path.exists(k_sidecar_path):
            cached_signature = open(k_sidecar_path).read().strip()

        if (
            self.output_adj_matrix_path is None
            or not os.path.exists(self.output_adj_matrix_path)
            or (signature is not None and cached_signature != signature)
        ):
            adj = _build_adj_matrix(
                self.knn_method,
                embeddings,
                is_similarity=self.is_similarity,
                output_adj_matrix_path=self.output_adj_matrix_path,
            )
            if k_sidecar_path is not None and signature is not None:
                with open(k_sidecar_path, "w") as f:
                    f.write(signature)
        else:
            adj = load_npz(self.output_adj_matrix_path)

        n_components, labels = _find_ccs(adj, self.threshold)

        print(
            f"Found {n_components} connected components from {len(embeddings)} images"
        )
        dup_indices = np.where(np.bincount(labels) > 1)[0]
        print(f"Found {len(dup_indices)} groups of duplicate images")

        df = self.df.with_columns(
            pl.Series(
                "Connected Component",
                np.array(labels, dtype=np.uint32),
                dtype=pl.Int32,
            )
        ).with_row_index("_index")

        print(
            f"Different values in 'Connected Component': {df.select(pl.col('Connected Component').n_unique())}"
        )

        print("Selecting first image in each connected component")
        df = select_first_in_group(df, "Connected Component", "Selected", "_index")
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        assert (
            df.select(pl.col("Selected")).sum().item() == n_components
        ), f"Number of selected images does not match number of connected components. Expected {n_components}, got {df.select(pl.col('Selected')).sum().item()}"

        if self.output_df_path is not None:
            if self.output_df_path.endswith(".csv"):
                df.write_csv(self.output_df_path)
            elif self.output_df_path.endswith(".parquet"):
                df.write_parquet(self.output_df_path)
            else:
                raise ValueError("Output dataframe format not supported")
