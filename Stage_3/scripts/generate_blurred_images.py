import argparse
import os
from pathlib import Path
from src.blur_text import MaskCustomWebDataset


def main():
    parser = argparse.ArgumentParser(description="Mask text regions in a WebDataset.")
    parser.add_argument(
        "--webdataset_dir",
        type=str,
        required=True,
        help="Path to the directory containing the WebDataset .tar files.",
    )
    parser.add_argument(
        "--bbox_info_dir",
        type=str,
        required=True,
        help="Path to the directory or file containing the bounding box information (in .parquet format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to store the output masked WebDataset.",
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="text-boxes",
        help="Column name in the bounding box parquet file that contains the polygons.",
    )

    args = parser.parse_args()

    # Create output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the appropriate class
    masker = MaskCustomWebDataset(
        webdataset_dir=args.webdataset_dir,
        bbox_info_dir=args.bbox_info_dir,
        output_dir=args.output_dir,
        column_name=args.column_name,
    )

    # Run the masking process
    masker()


if __name__ == "__main__":
    main()
