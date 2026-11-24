import cuml
import cudf
import numpy as np
from cuml.neighbors import NearestNeighbors
from .embedding_generator import EmbeddingReaderNewWDS


class CUMLKnn:
    def __init__(self, ref_embeddings, k: int, metric: str = "cosine") -> None:
        # Either an EmbeddingReaderNewWDS or an already-loaded array.
        if not isinstance(ref_embeddings, np.ndarray):
            ref_embeddings = ref_embeddings.load_embeddings(format="numpy")
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=k, metric=metric)
        self.knn.fit(ref_embeddings)

    def __call__(self, embeddings: np.ndarray, *args, **kwargs):
        return self.knn.kneighbors(embeddings)


def exact_cosine_distances(embeddings: np.ndarray, indices, chunk_size: int = 1024):
    """Exact cosine distance from each row of `embeddings` to its listed neighbours.

    Approximate indexes (IVF-PQ) return distances computed from quantised
    vectors, which are biased upwards, so the similarity threshold is applied
    to exact values recomputed for the returned neighbour pairs only.
    """
    indices = np.asarray(indices)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    distances = np.empty(indices.shape, dtype=np.float32)
    for start in range(0, len(normed), chunk_size):
        rows = normed[start : start + chunk_size]
        neighbours = normed[indices[start : start + chunk_size]]
        sims = np.einsum("id,ikd->ik", rows, neighbours)
        distances[start : start + chunk_size] = 1.0 - sims
    return distances


class CuMLSelfKnn:
    """k-NN over the query set itself.

    `algorithm="ivfpq"` builds the approximate inverted-file/product-quantised
    index; `algorithm="brute"` computes exact neighbours instead.
    `algo_params` is passed straight to cuML (nlist, nprobe, M, n_bits,
    usePrecomputedTables for ivfpq) and is ignored by the brute path. With
    ivfpq the index only selects the neighbours; the returned distances are
    exact cosine distances recomputed for those pairs, since the quantised
    ones are not comparable to a similarity threshold. If
    `algo_params["nlist"]` is None, it is computed at fit time from the
    number of embeddings (sqrt(N), capped at 1024), since it isn't known
    until `__call__` is given the data.
    """

    def __init__(
        self,
        k: int,
        metric: str = "cosine",
        algorithm: str = "ivfpq",
        algo_params: dict = None,
    ) -> None:
        self.k = k
        self.metric = metric
        self.algorithm = algorithm
        self.algo_params = algo_params
        self.knn = None
        if algorithm == "brute":
            self.knn = NearestNeighbors(
                n_neighbors=k, metric=metric, algorithm="brute"
            )

    def __call__(self, embeddings: np.ndarray, *args, **kwargs):
        if self.knn is None:
            algo_params = dict(self.algo_params or {})
            if algo_params.get("nlist") is None:
                n_samples = len(embeddings)
                algo_params["nlist"] = max(1, min(1024, int(np.sqrt(n_samples))))
            self.knn = NearestNeighbors(
                n_neighbors=self.k,
                metric=self.metric,
                algorithm=self.algorithm,
                algo_params=algo_params,
            )
        self.knn.fit(embeddings)
        distances, indices = self.knn.kneighbors(embeddings)
        if self.algorithm != "brute":
            indices = np.asarray(indices)
            distances = exact_cosine_distances(np.asarray(embeddings), indices)
        return distances, indices
