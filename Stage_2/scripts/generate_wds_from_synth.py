import argparse
import os
from pathlib import Path

import polars as pl
from tqdm import tqdm
import webdataset as wds

from src.prompts import load_concept_prompts


def convert_sd_output(args):
    """
    Convert a folder of images to a webdataset format.

    Args:
        args: Command-line arguments specifying input and output directories,
              prompt file, and other configuration options.
    """
    print(args)

    # Load the per-concept prompt list. Must match main_synth.py exactly, or
    # the file names built below won't line up with the images that were
    # actually generated.
    prompts = load_concept_prompts(args.prompts)

    # Create the output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Cumulative per-concept offset, for a collision-free ordinal_key.
    concept_lengths = [len(cps) for cps in prompts.values()]
    total_prompts = sum(concept_lengths)
    concept_offsets = [0]
    for length in concept_lengths[:-1]:
        concept_offsets.append(concept_offsets[-1] + length)

    for n in range(args.start_index, args.start_index + args.img_per_prompt):
        tar_file_path = os.path.join(args.output_dir, f"{n:06d}.tar")

        # Skip or overwrite existing tar files
        if Path(tar_file_path).exists():
            if not args.overwrite:
                print(f"File {tar_file_path} already exists. Skipping")
                continue
            else:
                print(f"File {tar_file_path} already exists. Overwriting")
                os.remove(tar_file_path)

        tar_writer = None
        rows = []

        # Generate images
        pbar = tqdm(
            total=sum([len(cps) for cps in prompts.values()]) * args.img_per_prompt,
            desc="Generating images",
        )

        for i, concept in enumerate(prompts):
            concept_prompts = prompts[concept]
            num_prompts = len(concept_prompts)
            for j in range(num_prompts):
                prompt = concept_prompts[j]
                fname = f"{i:05d}_{j}_{j}_{n:03d}"
                img_path = f"{args.input_dir}/{fname}.jpg"

                # Skip if image file does not exist
                if not Path(img_path).exists():
                    print(f"File {img_path} does not exist")
                    continue

                # Initialize tar writer if necessary
                if tar_writer is None:
                    tar_writer = wds.TarWriter(tar_file_path)

                # Generate ordinal key for ordering
                ordinal_key = n * total_prompts + concept_offsets[i] + j
                ordinal_key = f"{ordinal_key:09d}"

                # Write data to tar file
                data = {
                    "__key__": fname,
                    "jpg": open(img_path, "rb").read(),
                    "ordinal_key": ordinal_key,
                    "json": {
                        "prompt": prompt,
                        "negative_prompt": args.negative_prompt,
                        "i": i,
                        "j": j,
                        "n": n,
                        "concept": concept,
                    },
                }
                tar_writer.write(data)

                # Append metadata for Parquet file
                rows.append(
                    {
                        "prompt": prompt,
                        "negative_prompt": args.negative_prompt,
                        "i": i,
                        "j": j,
                        "n": n,
                        "concept": concept,
                        "jpg": img_path,
                        "key": fname,
                        "ordinal_key": ordinal_key,
                    }
                )

                pbar.update(1)

        # Write metadata to Parquet file if rows exist
        if rows:
            df = pl.DataFrame(rows)
            df.write_parquet(f"{args.output_dir}/{n:06d}.parquet")

        # Close tar writer if it was used
        if tar_writer is not None:
            tar_writer.close()

        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to webdataset format.")

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store output tar files.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="Path to JSON file containing prompts.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, drawing",
        help="Negative prompt for image generation.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for image generation.",
    )
    parser.add_argument(
        "--img-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing tar files."
    )

    args = parser.parse_args()
    convert_sd_output(args)
