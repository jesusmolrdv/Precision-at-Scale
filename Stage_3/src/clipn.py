import glob
import json
import os
from typing import List, Union

import polars as pl
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import webdataset as wds

from .classification import load_model


to_np = lambda x: x.detach().cpu().numpy()


def max_logit_score(logits):
    return to_np(torch.max(logits, -1)[0])


def msp_score(logits):
    prob = torch.softmax(logits, -1)
    return to_np(torch.max(prob, -1)[0])


def energy_score(logits):
    return to_np(torch.logsumexp(logits, -1))


def is_json_file(filename):
    try:
        with open(filename) as file:
            json.load(file)  # Try loading the file content as JSON
        return True
    except ValueError:
        return False


def load_queries_from_file(filename):
    if is_json_file(filename):
        with open(filename) as file:
            return json.load(file)  # Load and return queries from JSON file
    else:
        with open(filename) as file:
            return [
                line.rstrip() for line in file
            ]  # Return queries from a plain text file


class CLIPN:
    def __init__(
        self,
        pre_trained_model_path: str,
        classes: Union[str, List[str]],
        model_type: str = "ViT-B-16",
        cuda: bool = True,
        batch_size: int = 512,
    ) -> None:
        self.pre_trained_model_path = pre_trained_model_path
        if isinstance(classes, str):
            if os.path.exists(classes):
                classes = load_queries_from_file(classes)
            else:
                classes = [classes]
        self.classes = classes
        self.model_type = model_type
        self.cuda = cuda
        self.batch_size = batch_size

        print("Loading model")
        self.model, self.preprocess_train, self.preprocess_test = load_model(
            model_type,
            pre_trained_model_path,
            classes=self.classes,
            device="cuda" if self.cuda else "cpu",
        )
        if self.cuda:
            self.model = self.model.cuda()

        # Inference mode
        self.model.eval()
        self.model.fc_yes.requires_grad = False
        self.model.fc_no.requires_grad = False

    def process_batch(self, batch, jl_pool=None):
        if isinstance(batch, list):
            if jl_pool is not None:
                batch = jl_pool(delayed(self.preprocess_test)(b) for b in batch)
            else:
                batch = [self.preprocess_test(b) for b in batch]
            batch = torch.stack(batch)
        if self.cuda:
            batch = batch.cuda()
        with torch.no_grad():
            logits_yes, logits_no, _ = self.model(batch)
            idex = torch.argmax(logits_yes, -1).unsqueeze(-1)
            yesno = torch.cat([logits_yes.unsqueeze(-1), logits_no.unsqueeze(-1)], -1)
            yesno = torch.softmax(yesno, dim=-1)[:, :, 0]
            yesno_s = torch.gather(yesno, dim=1, index=idex).flatten()

        ood_logits = list(max_logit_score(logits_yes))
        ood_prob = list(msp_score(logits_yes))
        ood_energy = list(energy_score(logits_yes))

        return (
            ood_logits,
            ood_prob,
            ood_energy,
            list(to_np(yesno_s)),
            list(to_np((yesno * torch.softmax(logits_yes, -1)).sum(1))),
        )

    def process_wds(self, wds_dataset):
        batch = []
        keys = []

        jl_pool = Parallel(n_jobs=-1)

        for sample in wds_dataset:
            key, img, _json = sample
            batch.append(img)
            keys.append(key)
            if len(batch) == self.batch_size:
                yield keys, self.process_batch(batch, jl_pool)
                batch = []
                keys = []
        if len(batch) > 0:
            yield keys, self.process_batch(batch, jl_pool)


class CLIPNCuratorNewWDS:
    def __init__(
        self,
        ood_detector: CLIPN,
        df_paths: Union[str, List[str]],
        tar_dir: str,
        output_df_path: str,
        output_filtering_column: str = None,
        remove_columns: List[str] = [],
        append_blurred: bool = False,
    ):
        if isinstance(df_paths, str):
            # Apply glob pattern
            self.df_paths = sorted(list(glob.glob(df_paths)))
        else:
            self.df_paths = df_paths

        self.df_paths = sorted(self.df_paths)
        self.append_blurred = append_blurred
        if self.append_blurred:
            # Appends "_blurred" to the parent dir of every tarfile path, resolved
            # once here at construction rather than per-row.
            tar_dir = tar_dir.rstrip("/") + "_blurred"
        self.tar_dir = tar_dir
        self.output_df_path = output_df_path
        self.output_filtering_column = output_filtering_column
        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]
        self.remove_columns = remove_columns
        self.ood_detector = ood_detector

        def load_df(df_path):
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

        # "diagonal": web vs synthetic source parquets have disjoint schemas
        # (see Stage_3/src/deduplication.py / embedding_generator.py /
        # blur_text.py's own "diagonal" concats for the same reason).
        self.df = pl.concat(
            [load_df(df_path) for df_path in self.df_paths], how="diagonal"
        )
        self.df = self.df.drop(self.remove_columns)
        if isinstance(self.df, pl.LazyFrame):
            self.df = self.df.collect()

    def compute(self, *args, **kwargs) -> Union[pl.DataFrame, pl.LazyFrame]:
        _wds = self.get_wds()
        pbar = tqdm(total=len(self.df))

        test_keys = []
        all_ood_logits = []
        all_ood_prob = []
        all_ood_energy = []
        all_yesno_s = []
        all_yesno = []
        for batch in self.ood_detector.process_wds(_wds):
            keys, (ood_logits, ood_prob, ood_energy, yesno_s, yesno) = batch
            if len(test_keys) == 0:
                print(keys)
            test_keys.extend(keys)
            all_ood_logits.extend(ood_logits)
            all_ood_prob.extend(ood_prob)
            all_ood_energy.extend(ood_energy)
            all_yesno_s.extend(yesno_s)
            all_yesno.extend(yesno)
            pbar.update(len(keys))

        pbar.close()

        assert (
            test_keys == self.df.to_dict()["key"].to_list()
        ), "Keys are not in the same order"

        df = self.df.with_columns(
            [
                pl.Series("Logits", all_ood_logits),
                pl.Series("Prob", all_ood_prob),
                pl.Series("Energy", all_ood_energy),
                pl.Series("YesNo_S", all_yesno_s),
                pl.Series("YesNo", all_yesno),
            ]
        )

        return df

    def get_wds(self):
        tar_files = sorted(self.df.unique("tar_path").to_dict()["tar_path"])

        _wds = (
            wds.WebDataset(tar_files, shardshuffle=False)
            .decode("pil")
            .to_tuple("__key__", "jpg", "json")
        )

        return _wds

    def persist(self, df_new):
        if self.output_df_path is not None:
            if isinstance(df_new, pl.LazyFrame):
                df_new = df_new.collect()
            if self.output_df_path.endswith(".csv"):
                df_new.write_csv(self.output_df_path)
            elif self.output_df_path.endswith(".parquet"):
                df_new.write_parquet(self.output_df_path)
            else:
                raise NotImplementedError(
                    "output_df_path must end in .csv or .parquet"
                )
