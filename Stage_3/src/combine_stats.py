from collections import defaultdict
import os
from typing import Any, List, Union
import polars as pl
from loguru import logger
import funcy as fp

from .utils import (
    calculate_struct_column_differences,
    calculate_struct_column_minimums,
    load_pl_dataframe,
    write_pl_dataframe,
)

CLIPN_COLUMNS = ["Logits", "Prob", "Energy", "YesNo_S", "YesNo"]


def get_predefined_struct_columns(statistic):
    if "clipn" in statistic.lower():
        return CLIPN_COLUMNS


class StatisticReader:
    image_name_column = "image_name"

    def __init__(
        self, csv_creator, root_statistics_path, statistic, struct_columns=None
    ):
        self.csv_creator = csv_creator
        self.root_statistics_path = root_statistics_path
        self.statistic = statistic
        if struct_columns is None:
            struct_columns = get_predefined_struct_columns(statistic)
        elif len(struct_columns) == 0:
            struct_columns = None
        self.struct_columns = struct_columns

    @staticmethod
    def get_path(csv_creator, root_statistics_path, statistic):
        return os.path.join(
            root_statistics_path,
            f"{csv_creator}_{statistic}.parquet",
        )

    def read(
        self,
        lazy: bool = False,
        additional_columns: List[str] = [],
        debug: bool = False,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        logger.info(f"Reading {self.statistic} DataFrame for {self.csv_creator}")
        path = self.get_path(
            self.csv_creator, self.root_statistics_path, self.statistic
        )
        df = load_pl_dataframe(path, lazy=lazy)

        if self.struct_columns is not None:
            if debug:
                logger.info(f"Struct columns: {self.struct_columns}")
                logger.info(f"Additional columns: {additional_columns}")
                logger.info(f"Columns: {df.columns}")
            df = df.select(
                [self.image_name_column] + self.struct_columns + additional_columns
            )
            df = df.with_columns(pl.struct(self.struct_columns).alias(self.statistic))
            df = df.drop(self.struct_columns)

        return df


class DataFrameProcessor:
    def __init__(self, input_df_path, output_df_path, processor):
        self.input_df_path = input_df_path
        self.output_df_path = output_df_path
        self.processor = processor

    def __call__(
        self,
        return_df: bool = False,
        write_df: bool = True,
        input_df: pl.DataFrame = None,
    ) -> Any:
        if input_df is None:
            df = load_pl_dataframe(self.input_df_path)
        else:
            df = input_df
        print(df.head(5))
        df = self.processor(df)
        print(df.head(5))
        if write_df:
            write_pl_dataframe(df, self.output_df_path)
        if return_df:
            return df


class StatisticsOODPostProcessor(DataFrameProcessor):
    def __init__(self, input_df_path, output_df_path):
        processor = fp.compose(
            fp.partial(
                calculate_struct_column_differences,
                col1="detect_ood_clipn",
                col2="detect_ood_clipn_tmars",
                new_struct_name="diff_mars",
            ),
            fp.partial(
                calculate_struct_column_differences,
                col1="detect_ood_clipn_queries",
                col2="detect_ood_clipn_queries_tmars",
                new_struct_name="diff_queries_tmars",
            ),
            fp.partial(
                calculate_struct_column_minimums,
                col1="detect_ood_clipn",
                col2="detect_ood_clipn_queries",
                new_struct_name="min_clipn_queries",
            ),
        )
        super().__init__(input_df_path, output_df_path, processor)
        print(self.processor)
