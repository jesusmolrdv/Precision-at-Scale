import argparse
from src.clipn import CLIPN, CLIPNCuratorNewWDS


def main():
    parser = argparse.ArgumentParser(description="Run CLIPNCuratorNewWDS computation.")
    parser.add_argument(
        "--pre_trained_model_path",
        type=str,
        required=True,
        help="Path to pre-trained model.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Path to JSON file containing class names or a single class name.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Batch size for processing images."
    )
    parser.add_argument(
        "--df_paths",
        type=str,
        required=True,
        help="Path(s) to the dataset files (comma-separated if multiple).",
    )
    parser.add_argument(
        "--tar_dir", type=str, required=True, help="Directory containing the tar files."
    )
    parser.add_argument(
        "--output_df_path",
        type=str,
        required=True,
        help="Output path for the resulting DataFrame.",
    )
    parser.add_argument(
        "--append_blurred",
        action="store_true",
        help="Read images from the sibling '<tar_dir>_blurred/' directory instead of "
        "--tar_dir itself, for the _tmars variants. See Stage_3/README.md.",
    )
    args = parser.parse_args()

    # Load OOD detector
    ood_detector = CLIPN(
        pre_trained_model_path=args.pre_trained_model_path,
        classes=args.classes,
        batch_size=args.batch_size,
    )

    # Run curator
    curator = CLIPNCuratorNewWDS(
        ood_detector=ood_detector,
        df_paths=args.df_paths.split(","),
        tar_dir=args.tar_dir,
        output_df_path=args.output_df_path,
        append_blurred=args.append_blurred,
    )
    df_result = curator.compute()
    curator.persist(df_result)


if __name__ == "__main__":
    main()
