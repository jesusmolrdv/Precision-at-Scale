from copy import deepcopy
import os, sys
from pathlib import Path
import argparse

from tqdm import tqdm


from src.prompts import load_concept_prompts
from src.tensorrt.txt2img_pipeline import Txt2ImgPipeline
from src.tensorrt.utilities import DEFAULT_TENSORRT_ARGUMENTS


def main(args):

    print(args)

    # Load the per-concept prompt list (shared with generate_wds_from_synth.py)
    prompts = load_concept_prompts(args.prompts)

    print(f"Total number of concepts: {len(prompts)}")

    # Create required folders
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.engine_dir).mkdir(parents=True, exist_ok=True)
    Path(args.onnx_dir).mkdir(parents=True, exist_ok=True)

    pipeline = Txt2ImgPipeline(
        scheduler=args.scheduler,
        version=args.version,
        denoising_steps=args.denoising_steps,
        guidance_scale=args.guidance_scale,
    )

    assert args.model_img_size % 8 == 0, "--model-img-size must be a multiple of 8 (TensorRT engine-build requirement)"

    aux_tensorrt_arguments = deepcopy(DEFAULT_TENSORRT_ARGUMENTS)
    aux_tensorrt_arguments.update(
        {
            "engine_dir": args.engine_dir,
            "onnx_dir": args.onnx_dir,
            "opt_image_height": args.model_img_size,
            "opt_image_width": args.model_img_size,
        }
    )
    pipeline.loadEngines(**aux_tensorrt_arguments)
    pipeline.loadResources(args.model_img_size, args.model_img_size, 1, None)

    # Generate images. Each concept gets exactly its own prompts (no padding/
    # wraparound to some other concept's count), matching the paper's
    # |C| x N_Pro x N_Synth formula and generate_wds_from_synth.py's loop.
    pbar = tqdm(
        total=sum(len(cps) for cps in prompts.values()) * args.img_per_prompt,
        desc="Generating images",
    )
    for n in range(args.start_index, args.start_index + args.img_per_prompt):
        for i, concept in enumerate(prompts):
            concept_prompts = prompts[concept]
            for j, prompt in enumerate(concept_prompts):
                pipeline.infer(
                    [prompt],
                    [args.negative_prompt],
                    args.model_img_size,
                    args.model_img_size,
                    final_size=(args.final_size, args.final_size),
                    file_names=[f"{i:05d}_{j}_{j}_{n:03d}"],
                    output_dir=args.output_dir,
                )

                pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    """
    Example usage:
    python scripts/main_synth.py --prompts output/birds/prompts.json --output-dir output/birds/images/synth --img-per-prompt 35
    """
    parser = argparse.ArgumentParser(description="Options for Stable Diffusion Txt2Img")

    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="low quality, drawing")
    parser.add_argument("--model-img-size", type=int, default=512)
    parser.add_argument("--final-size", type=int, default=384)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--img-per-prompt", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)

    parser.add_argument("--version", type=str, default="2.1-base")
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="DDIM",
        choices=["PNDM", "LMSD", "DPM", "DDIM", "EulerA"],
        help="Scheduler for diffusion process",
    )
    parser.add_argument("--guidance-scale", type=float, default=7.5)

    parser.add_argument("--engine-dir", type=str, default="pretrained/txt2img/engines")
    parser.add_argument("--onnx-dir", type=str, default="pretrained/txt2img/onnx")

    args = parser.parse_args()
    main(args)
