import argparse
from src.pareto_front_filtering import (
    ParetoFrontFiltering,
    PercentageRemovedStopCriteria,
    RemainingImagesStopCriteria,
    KneeBasedStopCriteria,
)
import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Perform Pareto front filtering on an input dataset."
    )
    parser.add_argument(
        "--df_path",
        type=str,
        required=True,
        help="Path to the input dataframe (either .csv or .parquet).",
    )
    parser.add_argument(
        "--output_df_path",
        type=str,
        required=True,
        help="Path for the output filtered dataframe (either .csv or .parquet).",
    )
    parser.add_argument(
        "--flatten_columns",
        nargs="*",
        type=str,
        default=[],
        help="List of column names to flatten before filtering.",
    )
    parser.add_argument(
        "--remove_columns",
        nargs="*",
        type=str,
        default=[],
        help="List of column names to be removed from the dataframe.",
    )
    parser.add_argument(
        "--pareto_columns",
        nargs="+",
        type=str,
        required=True,
        help="List of column names used for Pareto front calculations.",
    )
    parser.add_argument(
        "--pareto_objectives",
        nargs="+",
        type=str,
        required=True,
        help="List of objectives ('min' or 'max') corresponding to each Pareto column.",
    )
    parser.add_argument(
        "--stop_criteria",
        type=str,
        required=True,
        choices=[
            "PercentageRemovedStopCriteria",
            "RemainingImagesStopCriteria",
            "KneeBasedStopCriteria",
        ],
        help="Type of stop criteria to use.",
    )
    parser.add_argument(
        "--stop_criteria_args",
        type=str,
        help="Additional arguments for the stop criteria, provided as a YAML string.",
    )
    parser.add_argument(
        "--output_filtering_column",
        type=str,
        default="Selected",
        help="Name of the output column indicating the selection status.",
    )

    args = parser.parse_args()

    # Parse stop criteria arguments
    stop_criteria_args = (
        yaml.safe_load(args.stop_criteria_args) if args.stop_criteria_args else {}
    )

    # Initialize stop criteria
    if args.stop_criteria == "PercentageRemovedStopCriteria":
        stop_criteria = PercentageRemovedStopCriteria(**stop_criteria_args)
    elif args.stop_criteria == "RemainingImagesStopCriteria":
        stop_criteria = RemainingImagesStopCriteria(**stop_criteria_args)
    elif args.stop_criteria == "KneeBasedStopCriteria":
        stop_criteria = KneeBasedStopCriteria(**stop_criteria_args)

    # Initialize ParetoFrontFiltering
    pareto_filtering = ParetoFrontFiltering(
        df_path=args.df_path,
        output_df_path=args.output_df_path,
        pareto_columns=args.pareto_columns,
        pareto_objectives=args.pareto_objectives,
        stop_criteria=stop_criteria,
        flatten_columns=args.flatten_columns,
        remove_columns=args.remove_columns,
        output_filtering_column=args.output_filtering_column,
    )

    # Run Pareto filtering
    pareto_filtering()


if __name__ == "__main__":
    main()
