import os
from typing import Union
import torch
from torchvision import transforms
from PIL import Image
import tarfile
import numpy as np
import polars as pl
from tqdm import tqdm
from io import BytesIO
from glob import glob

# Normalization for embedding models
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

# Transformation used in image preprocessing
skew_320 = transforms.Compose(
    [
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ]
)


class CustomWebDataset:
    _wds = None

    def __init__(
        self,
        images_dir=None,
        df_path=None,
        filtered_column=None,
        idx_column=None,
        tar_dir=None,
    ):
        self.images_dir = images_dir
        self.df_path = df_path
        self.filtered_column = filtered_column
        self.idx_column = idx_column
        self.tar_dir = tar_dir

    def load_df(self, only_downloaded=True, return_lazy=False, apply_filter=True):

        if return_lazy:
            print("WARNING: returning lazy dataframe is not supported yet")
            return_lazy = False

        # Load all the parquet files matching the pattern
        if isinstance(self.df_path, str):
            _df_paths = sorted(glob(self.df_path))
        else:
            _df_paths = sorted(self.df_path)

        print(_df_paths)

        if self.tar_dir is None:
            dfs = [
                pl.read_parquet(df_path).with_columns(
                    pl.lit(os.path.basename(df_path)).alias("df_name"),
                    pl.lit(df_path.replace(".parquet", ".tar")).alias("tarfile"),
                )
                for df_path in _df_paths
            ]
        else:
            dfs = [
                pl.read_parquet(df_path).with_columns(
                    pl.lit(os.path.basename(df_path)).alias("df_name"),
                    pl.lit(
                        os.path.join(
                            self.tar_dir,
                            os.path.basename(df_path).replace(".parquet", ".tar"),
                        )
                    ).alias("tarfile"),
                )
                for df_path in _df_paths
            ]
        # "diagonal" since a unified input can combine chunks from sources
        # with disjoint schemas (e.g. web vs synthetic) — see
        # Stage_2/scripts/combine_wds_sources.py and
        # Stage_3/src/deduplication.py's identically-motivated concat.
        df = pl.concat(dfs, how="diagonal").sort("ordinal_key")
        if only_downloaded:
            # If the column "downloaded" is present, filter the rows with downloaded=True
            if "downloaded" in df.columns:
                df = df.filter(pl.col("downloaded") == True)
        if apply_filter and self.filtered_column is not None:
            df = df.filter(pl.col(self.filtered_column) == True)

        if return_lazy:
            return df.lazy()
        else:
            return df

    def get_wds(self):
        if self._wds is None:
            from webdataset import WebDataset

            # Get list of unique tarfiles in the dataset from the polars dataframe
            tarfiles = sorted(
                self.load_df(only_downloaded=True)["tarfile"]
                .unique()
                .to_list()
            )

            print("TAR FILES")
            print(tarfiles)

            self._wds = (
                WebDataset(tarfiles, shardshuffle=False)
                .decode("pil")
                .to_tuple("__key__", "jpg", "json")
            )
        return self._wds


class EmbeddingGeneratorNewWDS:
    def __init__(
        self,
        embedding_dir: str,
        embedding_size: int,
        embedding_model: torch.nn.Module,
        dataset: Union[str, "CustomWebDataset"],
        cuda: bool = True,
        batch_size: int = 128,
        chunk_by_n: bool = True,
        transform=skew_320,
        save_metadata: bool = True,
        **kwargs,
    ):
        import webdataset as wds  # Lazy import to avoid unnecessary dependency errors

        if isinstance(dataset, str):
            dataset = CustomWebDataset(None, dataset, None)
        self.embedding_dir = embedding_dir
        self.embedding_size = embedding_size
        self.embedding_model = embedding_model
        self.dataset = dataset
        self.cuda = cuda
        self.batch_size = batch_size
        self.transform = transform
        self.save_metadata = save_metadata
        self.chunk_by_n = chunk_by_n
        self.initialized = False

        # Create embedding directory if it does not exist
        os.makedirs(self.embedding_dir, exist_ok=True)

        # Load any additional attributes provided
        for key, value in kwargs.items():
            setattr(self, key, value)

    def init_model(self):
        if not self.initialized:
            self.embedding_model = self.embedding_model()
            if getattr(self, "force_fc_identity", False):
                self.embedding_model.fc = torch.nn.Identity()
            self.initialized = True
        if self.cuda:
            self.embedding_model = self.embedding_model.cuda()
        self.embedding_model.eval()

    def build_embeddings(self):
        """Generates embeddings for the dataset."""
        self.init_model()
        df = self.dataset.load_df(only_downloaded=True, return_lazy=False)

        pbar = tqdm(
            total=len(df), desc="Generating embeddings", unit="image", leave=False
        )
        imgs = []
        img_keys = []
        embeddings_by_key = {}

        def process_batch(batch, batch_keys):
            batch = torch.stack(batch)
            if self.cuda:
                batch = batch.cuda()
            with torch.no_grad():
                embeddings = self.forward(batch)
            embeddings = embeddings.cpu()  # Move to CPU to save memory
            for k, e in zip(batch_keys, embeddings):
                embeddings_by_key[k] = e

        # Extracting keys from dataframe
        keys = df["key"].to_list()

        # get_wds() iterates tar files in sorted-tarfile-name order (then
        # physical/insertion order within each tar) — for a combined
        # multi-source dataset that's a different order than df's global
        # sort by "ordinal_key" (each source's ordinal_key is only
        # meaningful within that source, so interleaving sources by it
        # doesn't line up with per-tar iteration order). Match samples to
        # df rows by key instead of assuming the two orderings coincide.
        for sample in self.dataset.get_wds():
            key, image, metadata = sample
            imgs.append(self.transform(image))
            img_keys.append(key)
            if len(imgs) == self.batch_size:
                process_batch(imgs, img_keys)
                imgs = []
                img_keys = []
            pbar.update(1)

        # Process remaining images if any
        if len(imgs) > 0:
            process_batch(imgs, img_keys)

        all_embeddings = torch.stack([embeddings_by_key[k] for k in keys])

        # Persist embeddings based on unique "n" values or create a single chunk
        if self.chunk_by_n:
            unique_ns = df["n"].unique().to_list()
            for n in unique_ns:
                n_mask = (df["n"] == n).to_list()
                n_embeddings = all_embeddings[n_mask]
                self.persist_chunk(
                    slice=df.filter(pl.col("n") == n), all_embeddings=n_embeddings, i=n
                )
        else:
            if "n" in df.columns:
                unique_ns = df["n"].unique().to_list()
                n = unique_ns[0] if len(unique_ns) == 1 else 0
                self.persist_chunk(slice=df, all_embeddings=all_embeddings, i=n)
            else:
                self.persist_chunk(slice=df, all_embeddings=all_embeddings, i=0)

    def persist_chunk(self, slice, all_embeddings, i, chunk_format_str="chunk_{:06d}"):
        """Saves a chunk of embeddings to disk."""
        if not isinstance(all_embeddings, torch.Tensor):
            all_embeddings = torch.stack(all_embeddings)
        torch.save(
            all_embeddings,
            self.embedding_dir + "/" + chunk_format_str.format(i) + ".pt",
        )
        if self.save_metadata:
            slice.write_parquet(
                self.embedding_dir + "/" + chunk_format_str.format(i) + ".parquet"
            )

    def _read_metadata_chunk(self, chunk_num, lazy, chunk_format_str):
        """Reads one persisted chunk's metadata parquet from disk."""
        path = self.embedding_dir + "/" + chunk_format_str.format(chunk_num) + ".parquet"
        return pl.scan_parquet(path) if lazy else pl.read_parquet(path)

    def _read_embeddings_chunk(self, chunk_num, format, chunk_format_str):
        """Reads one persisted chunk's embeddings .pt from disk."""
        path = self.embedding_dir + "/" + chunk_format_str.format(chunk_num) + ".pt"
        embeddings = torch.load(path)
        return embeddings.numpy() if format == "numpy" else embeddings

    def load_metadata(
        self, chunk_num=None, lazy=False, chunk_format_str="chunk_{:06d}"
    ):
        """Loads metadata from disk."""
        if self.chunk_by_n:
            unique_ns = sorted(
                self.dataset.load_df()["n"].unique().to_list()
            )
            all_metadata = [
                self._read_metadata_chunk(n, lazy, chunk_format_str) for n in unique_ns
            ]
            return pl.concat(all_metadata)
        return self._read_metadata_chunk(chunk_num, lazy, chunk_format_str)

    def load_embeddings(
        self, chunk_num: int = None, format="numpy", chunk_format_str="chunk_{:06d}"
    ):
        """Loads embeddings from disk."""
        if self.chunk_by_n:
            unique_ns = sorted(
                self.dataset.load_df()["n"].unique().to_list()
            )
            all_embeddings = [
                self._read_embeddings_chunk(n, format, chunk_format_str)
                for n in unique_ns
            ]
            return np.concatenate(all_embeddings)
        return self._read_embeddings_chunk(chunk_num, format, chunk_format_str)

    def forward(self, x):
        """Forward pass for the embedding model."""
        return self.embedding_model(x)


class EmbeddingReaderNewWDS(EmbeddingGeneratorNewWDS):
    def __init__(
        self,
        embedding_dir: str,
        embedding_size: int,
        filtering_df_path: str = None,
        embedding_model: torch.nn.Module = None,
        dataset: Union[CustomWebDataset, str] = None,
        cuda: bool = True,
        batch_size: int = 128,
        transform=skew_320,
        save_metadata: bool = True,
        chunk_by_n: bool = True,
        metadata_key_column: str = "key",
        filtering_df_key_column: str = "key",
        **kwargs,
    ):
        assert (
            save_metadata == True
        ), "EmbeddingReaderNewWDS requires metadata to be saved."
        assert (
            embedding_model is None
        ), "EmbeddingReaderNewWDS does not require an embedding model."
        super().__init__(
            embedding_dir=embedding_dir,
            embedding_size=embedding_size,
            embedding_model=embedding_model,
            dataset=dataset,
            cuda=cuda,
            batch_size=batch_size,
            transform=transform,
            save_metadata=save_metadata,
            chunk_by_n=chunk_by_n,
            **kwargs,
        )
        self.filtering_df_path = filtering_df_path
        if self.filtering_df_path is not None:
            if self.filtering_df_path.endswith(".csv"):
                self.filtering_df = pl.scan_csv(self.filtering_df_path)
            elif self.filtering_df_path.endswith(".parquet"):
                self.filtering_df = pl.read_parquet(self.filtering_df_path)
        else:
            self.filtering_df = None

        self.metadata_key_column = metadata_key_column
        self.filtering_df_key_column = filtering_df_key_column

    def init_model(self):
        # Warn user that this is not the intended use case
        print(
            "WARNING: EmbeddingReaderNewWDS is not intended for embedding generation. Use EmbeddingGeneratorNewWDS instead."
        )

    def build_embeddings(self):
        # Warn user that this is not the intended use case
        print(
            "WARNING: EmbeddingReaderNewWDS is not intended for embedding generation. Use EmbeddingGeneratorNewWDS instead."
        )

    def persist_chunk(self, slice, all_embeddings, i, chunk_format_str="chunk_{:06d}"):
        # Warn user that this is not the intended use case
        print(
            "WARNING: EmbeddingReaderNewWDS is not intended for embedding generation. Use EmbeddingGeneratorNewWDS instead."
        )

    def get_filtered_embeddings(
        self, filtering_df: pl.DataFrame = None, format="numpy"
    ):
        metadata = self.load_metadata(lazy=False).with_row_count("general_index")
        embeddings = self.load_embeddings(format=format)

        filtering_df = self.filtering_df if filtering_df is None else filtering_df

        if filtering_df is not None:
            metadata = filtering_df.join(
                metadata,
                left_on=self.filtering_df_key_column,
                right_on=self.metadata_key_column,
                how="left",
            )

            valid_indices = (
                metadata.filter(pl.col("general_index") >= 0)
                .select(pl.col("general_index"))["general_index"]
                .to_list()
            )

            return embeddings[valid_indices]
        else:
            return embeddings
