import glob
import os
from pathlib import Path
from typing import Any, List, Union
import cv2
import numpy as np
import polars as pl
from tqdm import tqdm

from .embedding_generator import CustomWebDataset


def mask_text_poly(image_path, polygons, output_path=None, img=None):
    # Load the image
    if img is None:
        img = cv2.imread(image_path)

    # For each polygon
    for polygon in polygons:
        # Convert polygon to numpy array and reshape it
        poly_np = np.array(polygon, dtype=np.int32)
        # Recompute the polygons taking into account that they were computed on a resized image to a 512 x 512 resolution
        # While keeping the aspect ratio. It mask take into account the zero padding added to keep the aspect ratio and the image in the center
        if img.shape[0] > img.shape[1]:
            # The image is taller than wider
            # The height is 512
            # The width is 512 * img.shape[1] / img.shape[0]
            used_height = 512
            used_width = int(512 * img.shape[1] / img.shape[0])
            # The padding is 512 - used_width
            padding = 512 - used_width
            # The padding is split in two
            padding_left = padding // 2
            # The polygon is translated by padding_left
            poly_np[:, 0] -= padding_left
            poly_np = poly_np / (512 / img.shape[0])
        else:
            # The image is wider than taller
            # The width is 512
            # The height is 512 * img.shape[0] / img.shape[1]
            used_width = 512
            used_height = int(512 * img.shape[0] / img.shape[1])
            # The padding is 512 - used_height
            padding = 512 - used_height
            # The padding is split in two
            padding_top = padding // 2
            # The polygon is translated by padding_top
            poly_np[:, 1] -= padding_top
            poly_np = poly_np / (512 / img.shape[1])

        poly_np = poly_np.astype(np.int32)

        # Create a mask for the polygon
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [poly_np], color=(255, 255, 255))

        # Compute the mean color of the polygon region
        # Convert mask to 2D array
        mask_2d = mask[:, :, 0]
        mean_color = cv2.mean(img, mask=mask_2d)[:3]

        img = cv2.fillPoly(img, [poly_np], color=mean_color)

    if output_path is not None:
        # Save the result to the specified output path
        cv2.imwrite(output_path, img)
    return img


class MaskCustomWebDataset:
    def __init__(
        self,
        webdataset_dir: str,
        bbox_info_dir: Union[str, List[str]],
        output_dir: str,
        column_name: str = "text-boxes",
    ):
        self.short_size = 512

        self.webdataset_dir = webdataset_dir
        self.bbox_info_dir = bbox_info_dir
        self.output_dir = output_dir
        self.column_name = column_name

        print(self.bbox_info_dir)

        self.tar_files = glob.glob(os.path.join(self.webdataset_dir, "*.tar"))
        self.tar_files = sorted(self.tar_files)
        self.parquet_files = []
        for tar_file in self.tar_files:
            self.parquet_files.append(tar_file.replace(".tar", ".parquet"))
        for i in range(len(self.parquet_files)):
            self.parquet_files[i] = pl.scan_parquet(self.parquet_files[i]).with_columns(
                pl.lit(self.tar_files[i]).alias("tar_file")
            )
        # "diagonal": web vs synthetic source parquets have disjoint schemas
        # (see Stage_3/src/deduplication.py / embedding_generator.py's own
        # "diagonal" concats for the same reason) -- a plain concat crashes
        # combining them.
        self.df_parquet = pl.concat(self.parquet_files, how="diagonal").collect()

        if not isinstance(self.bbox_info_dir, str) and len(self.bbox_info_dir) > 1:
            self.df_bboxes = []
            for bbox_df_path in self.bbox_info_dir:
                self.df_bboxes.append(pl.read_parquet(bbox_df_path))
                print(f"Reading {bbox_df_path} with {len(self.df_bboxes[-1])} rows")
            self.df_bboxes = pl.concat(self.df_bboxes)
        elif os.path.isdir(self.bbox_info_dir):
            self.bbox_files = sorted(glob.glob(os.path.join(self.bbox_info_dir, "*.parquet")))
            print(f"Reading bbox info from {len(self.bbox_files)} file(s) in directory {self.bbox_info_dir}")
            self.df_bboxes = pl.concat([pl.read_parquet(f) for f in self.bbox_files])
        else:
            print(f"Reading bbox info from file {self.bbox_info_dir}")
            self.df_bboxes = pl.read_parquet(self.bbox_info_dir)

        print(f"Reading {self.bbox_info_dir} with {len(self.df_bboxes)} rows")

        assert len(self.df_parquet) == len(
            self.df_bboxes
        ), "The number of rows in the parquet files and in the bbox files must be the same. Got {} and {}".format(
            len(self.df_parquet), len(self.df_bboxes)
        )

        self.df_parquet = self.df_parquet.with_columns(
            pl.Series(self.column_name, self.df_bboxes[self.column_name].to_list())
        )

        import webdataset as wds

        self._wds = (
            wds.WebDataset(self.tar_files, shardshuffle=False)
            .decode("pil")
            .to_tuple("__key__", "jpg", "json")
        )

    def _mask_image(self, img, polygons):
        return mask_text_poly(
            image_path=None, polygons=polygons, output_path=None, img=img
        )

    def __call__(self, *args, **kwargs):
        import webdataset as wds

        with tqdm(total=len(self.df_parquet)) as pbar:

            all_polygons = self.df_parquet[self.column_name].to_list()
            all_tar_files = self.df_parquet["tar_file"].to_list()

            tar_writers = {}

            for idx, sample in enumerate(self._wds):
                key, img, json = sample
                img = np.array(img)
                polygons = all_polygons[idx]
                img = self._mask_image(img, polygons)

                tar_file = all_tar_files[idx]
                output_tar_file = os.path.join(
                    self.output_dir, os.path.basename(tar_file)
                )

                if tar_file not in tar_writers:
                    tar_writers[tar_file] = wds.TarWriter(output_tar_file)

                json["text-boxes"] = polygons

                sample = {"__key__": key, "jpg": img, "json": json}
                tar_writers[tar_file].write(sample)

                pbar.update(1)

            for tar_writer in tar_writers.values():
                tar_writer.close()
