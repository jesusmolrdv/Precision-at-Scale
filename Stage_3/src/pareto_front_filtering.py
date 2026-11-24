from abc import ABC, abstractmethod
import numpy as np
import polars as pl
from typing import Any, List
from kneed import KneeLocator
from paretoset import paretoset
from .utils import flatten_struct_column
from tqdm import tqdm


class StopCriteria(ABC):
    @abstractmethod
    def update(self, df, position):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def get_position(self):
        pass


class PercentageRemovedStopCriteria(StopCriteria):
    def __init__(self, threshold_percentage: float):
        self.threshold_percentage = threshold_percentage
        self.initial_length = None
        self.current_length = None

    def update(self, df, position):
        if self.initial_length is None:
            self.initial_length = len(df)
        self.current_length = len(df[position == 0])
        self.position = position

    def stop(self):
        if self.initial_length is None:
            return False
        removed_percentage = (
            (self.initial_length - self.current_length) / self.initial_length
        ) * 100
        return removed_percentage > self.threshold_percentage

    def get_position(self):
        return self.position


class RemainingImagesStopCriteria(StopCriteria):
    def __init__(self, threshold_number: int):
        self.threshold_number = threshold_number

    def update(self, df, position):
        self.current_length = np.sum([position == 0])
        self.position = position

    def stop(self):
        return self.current_length <= self.threshold_number

    def get_position(self):
        return self.position


class KneeBasedStopCriteria(StopCriteria):
    def __init__(
        self,
        columns: List[str],
        curves: str,
        directions: str,
        auxiliar_stop_criteria: StopCriteria = None,
    ):
        self.columns = columns
        self.curves = curves
        self.directions = directions
        assert len(columns) == len(curves) == len(directions)
        self.auxiliar_stop_criteria = (
            auxiliar_stop_criteria
            if auxiliar_stop_criteria is not None
            else RemainingImagesStopCriteria(0)
        )

    def update(self, df, position):
        self.position = position
        self.df = df
        self.auxiliar_stop_criteria.update(df, position)

    def stop(self):
        return self.auxiliar_stop_criteria.stop()

    def get_position(self):
        # From pandas to polars
        df = pl.from_pandas(self.df)

        df = df.select(self.columns).with_columns(
            pl.Series("ParetoPosition", self.position)
        )
        stats_per_position = (
            df.group_by("ParetoPosition")
            .agg(
                [pl.mean(c).alias(c) for c in self.columns]
                + [pl.count("ParetoPosition").alias("Count")]
            )
            .filter(pl.col("ParetoPosition") > 0)
            .sort("ParetoPosition")
        )
        stats_per_position = stats_per_position.with_columns(
            pl.col("Count").cum_sum().alias("CumulativeCount")
        )

        stats_per_position = stats_per_position.to_pandas()
        stats_per_position["ParetoPosition"] = stats_per_position[
            "ParetoPosition"
        ].astype(int)
        stats_per_position = stats_per_position.set_index("ParetoPosition")
        stats_per_position = stats_per_position.reset_index()

        max_knee = -1

        x = stats_per_position["CumulativeCount"].to_numpy()

        for column, curve, direction in zip(self.columns, self.curves, self.directions):
            y = stats_per_position[column].to_numpy()

            multiplier = 1 if direction == "increasing" else -1

            for i in range(1, len(y)):
                if y[i] * multiplier < y[i - 1] * multiplier:
                    y[i] = y[i - 1]

            kl = KneeLocator(
                x, y, curve=curve, direction=direction, interp_method="polynomial"
            )

            print(f"Knee for {column} is {kl.knee}")

            if kl.knee is not None:
                max_knee = max(kl.knee, max_knee)

        # Translate the knee from CumulativeCount to ParetoPosition
        knee = stats_per_position[stats_per_position["CumulativeCount"] == max_knee][
            "ParetoPosition"
        ].values[0]

        # Force of the positions above the knee to be 0
        print(f"Knee is {knee}. Setting positions above the knee to 0")
        self.position[self.position > knee] = 0

        return self.position


class ParetoFrontFiltering:
    """
    This class implements Pareto front filtering on a given dataframe for image curation.

    Attributes:
        df_path (str): Path to the input dataframe.
        output_df_path (str): Path for the output filtered dataframe.
        pareto_columns (List[str]): List of column names used for Pareto front calculations.
        pareto_objectives (List[str]): List of objectives (e.g., 'min' or 'max') corresponding to each Pareto column.
        stop_criteria (StopCriteria): An instance of StopCriteria to determine when the filtering process should stop.
        flatten_columns (List[str]): List of column names that need flattening before filtering.
        remove_columns (List[str]): List of column names to be removed from the dataframe.
        output_filtering_column (str): Name of the column indicating the selection status in the output dataframe.

    Methods:
        filter(*args, **kwargs) -> List[int]: Performs the Pareto front filtering on the dataframe.
    """

    def __init__(
        self,
        df_path: str,
        output_df_path: str,
        pareto_columns: List[str],
        pareto_objectives: List[str],
        stop_criteria: StopCriteria,
        flatten_columns: List[str] = [],
        remove_columns: List[str] = [],
        output_filtering_column: str = "Selected",
    ) -> None:
        assert len(pareto_columns) == len(pareto_objectives)
        self.df_path = df_path
        self.output_df_path = output_df_path
        self.output_filtering_column = output_filtering_column

        if isinstance(remove_columns, str):
            remove_columns = [remove_columns]
        self.remove_columns = remove_columns

        if self.df_path.endswith(".csv"):
            self.df = pl.read_csv(self.df_path)
        elif self.df_path.endswith(".parquet"):
            self.df = pl.read_parquet(self.df_path)
        if "downloaded" in self.df.columns:
            self.df = self.df.filter(pl.col("downloaded") == True)
        if output_filtering_column in self.df.columns:
            self.df = self.df.filter(pl.col(output_filtering_column) == True)
        self.df = self.df.drop(self.remove_columns)
        if isinstance(self.df, pl.LazyFrame):
            self.df = self.df.collect()

        self.pareto_columns = pareto_columns
        self.pareto_objectives = pareto_objectives
        self.stop_criteria = stop_criteria
        self.flatten_columns = flatten_columns

    def filter(self, *args, **kwargs) -> List[int]:
        df = self.df

        for column in self.flatten_columns:
            df = flatten_struct_column(df, column)

        df = df.select(self.pareto_columns).to_pandas()

        if isinstance(self.stop_criteria, KneeBasedStopCriteria):
            pbar = tqdm(total=len(df))
        elif isinstance(self.stop_criteria, PercentageRemovedStopCriteria):
            pbar = tqdm(total=int(len(df) * self.stop_criteria.threshold_percentage / 100))
        elif isinstance(self.stop_criteria, RemainingImagesStopCriteria):
            pbar = tqdm(total=len(df) - self.stop_criteria.threshold_number)
        else:
            pbar = tqdm()

        position = np.zeros(len(df), dtype=np.int32)
        mask = paretoset(df, sense=self.pareto_objectives)
        position[mask] = 1

        self.stop_criteria.update(df, position)
        pbar.update(np.sum(mask))

        iteration = 0
        while not self.stop_criteria.stop():
            iteration += 1
            print(f"Running iteration {iteration}")
            df_remaining = df[~mask]
            mask_remaining = paretoset(df_remaining, sense=self.pareto_objectives)
            print(f"Found {np.sum(mask_remaining)} non-relevant images")
            print(f"Remaining images: {len(df_remaining)}")
            position[df_remaining.index[mask_remaining]] = iteration + 1

            mask[df_remaining.index[mask_remaining]] = True
            self.stop_criteria.update(df, position)
            pbar.update(np.sum(mask_remaining))
        pbar.close()

        print(f"Found {np.sum(mask)} non-relevant images from {len(df)} images")
        position = self.stop_criteria.get_position()
        print(f"Found {np.sum(position > 0)} non-relevant images")
        # Recompute the mask
        mask = position > 0

        # Add column with the Pareto front position to the dataframe
        self.df = self.df.with_columns(pl.Series("ParetoPosition", position))

        # Return indices of the Pareto front
        return list(df.index[~mask])

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        filtered_indices = self.filter(*args, **kwds)
        selected = np.zeros(len(self.df), dtype=bool)
        selected[filtered_indices] = True
        self.df = self.df.with_columns(
            pl.Series(self.output_filtering_column, selected)
        )
        print(
            f"Found {len(filtered_indices)} relevant images from {len(self.df)} images"
        )
        self.persist()

    def persist(self):
        if self.output_df_path is not None:
            if not isinstance(self.df, pl.DataFrame):
                self.df = self.df.collect()
            print(self.df)
            if self.output_df_path.endswith(".csv"):
                self.df.write_csv(self.output_df_path)
            elif self.output_df_path.endswith(".parquet"):
                self.df.write_parquet(self.output_df_path)
