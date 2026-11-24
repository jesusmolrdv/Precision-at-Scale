"""
Generate detailed text prompts using LlamaCpp language models.

This script processes concepts from various file formats and generates corresponding prompts
using different LLM architectures (Mixtral, Mistral, LLaMA2). It includes built-in retry logic,
progress tracking, and periodic saving of results.

Models supported:
- Mixtral 8x7B (default)
- Mistral 7B
- LLaMA2 13B

The script expects:
1. A template file with the prompt structure
2. A file containing concepts (JSON/YAML/TXT)
3. Domain and category descriptions
4. Output path for generated prompts


Usage example:

python scripts/main_prompts.py --concepts ../Stage_1/output/birds/valid_concepts.json --domain birds --category-desc "bird species" --output output/birds/prompts.json --num-prompts 5
"""

import argparse
import json
import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path

from tqdm import tqdm
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from collections import defaultdict


def build_configs(root_dir: str) -> Dict:
    """
    Build configuration dictionary for supported LLM models.

    Args:
        root_dir: Root directory containing model files

    Returns:
        Dictionary with model configurations

    Raises:
        ValueError: If root directory doesn't exist
    """
    if not Path(root_dir).exists():
        raise ValueError(f"Root directory not found: {root_dir}")

    return {
        "mixtral": {
            "model": os.path.join(
                root_dir, "models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"
            ),
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "max_tokens": 2048,
        },
        "mistral": {
            "model": os.path.join(
                root_dir, "models/mistral-7b-instruct-v0.1.Q5_K_M.gguf"
            ),
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "max_tokens": 2048,
        },
        "llama2-13b": {
            "model": os.path.join(root_dir, "models/llama-2-13b-chat.Q5_K_M.gguf"),
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "max_tokens": 2048,
        },
    }


def read_concepts(file_path: str) -> List[str]:
    """
    Read and parse concepts from supported file formats.

    Args:
        file_path: Path to concepts file (JSON/YAML/TXT)

    Returns:
        List of cleaned concept strings

    Raises:
        ValueError: If file format is not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Concepts file not found: {file_path}")

    if path.suffix == ".json":
        concepts = json.loads(path.read_text())
        if isinstance(concepts, dict):
            concepts = list(concepts.values())
    elif path.suffix in {".yaml", ".yml"}:
        concepts = yaml.safe_load(path.read_text())
    elif path.suffix == ".txt":
        concepts = path.read_text().splitlines()
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return [c.strip() for c in concepts if c and c.strip()]


def save_prompts(prompts: Dict, output_path: str) -> None:
    """Save generated prompts to JSON file."""
    with open(output_path, "w") as f:
        json.dump(prompts, f, indent=4)


def setup_llm_chain(args: argparse.Namespace) -> tuple:
    """
    Set up LLM chain with template and parser.

    Returns:
        Tuple of (LLM chain, template)
    """
    config = build_configs(args.root_dir)[args.model]
    config.update({"n_ctx": 512, "max_tokens": 512})

    llm = LlamaCpp(
        model_path=config["model"],
        n_gpu_layers=config["n_gpu_layers"],
        n_ctx=config["n_ctx"],
        f16_kv=True,
        verbose=False,
        grammar_path=Path(args.root_dir) / "grammars/json_prompt.gbnf",
        n_batch=config["n_ctx"],
        max_tokens=config["max_tokens"],
        streaming=False,
    )

    template = Path(args.template).read_text()

    return (
        PromptTemplate(
            template=template, input_variables=["concept", "domain", "category_desc"]
        )
        | llm
        | JsonOutputParser()
    ), template


def generate_prompt(
    chain, concept: str, args: argparse.Namespace, retries: int = 5
) -> Optional[Dict]:
    """Generate a single prompt with retry logic."""
    for _ in range(retries):
        try:
            output = chain.invoke(
                {
                    "concept": concept,
                    "domain": args.domain,
                    "category_desc": args.category_desc,
                }
            )
            if output["image_caption"].strip():
                return output
            raise ValueError("Empty prompt generated")
        except Exception as e:
            print(f"Error generating prompt for {concept}: {e}")
    return None


def main(args: argparse.Namespace) -> None:
    """Generate prompts using configured LLM chain."""
    concepts = read_concepts(args.concepts)
    print(f"Processing {len(concepts)} concepts from {args.concepts}")

    llm_chain, _ = setup_llm_chain(args)
    generated_prompts = defaultdict(list)
    total_iterations = len(concepts) * args.num_prompts

    num_generated = 0
    with tqdm(total=total_iterations, desc="Generating prompts") as pbar:
        for _ in range(args.num_prompts):
            for concept in concepts:
                if output := generate_prompt(
                    llm_chain, concept, args, args.num_retries
                ):
                    generated_prompts[concept].append(output)
                    num_generated += 1

                pbar.update(1)
                if pbar.n % args.save_every == 0:
                    save_prompts(generated_prompts, args.output)
                    print(f"\nSaved {pbar.n} prompts to {args.output}")

    save_prompts(generated_prompts, args.output)
    print(f"\nCompleted! Generated {num_generated}/{total_iterations} prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate prompts using LlamaCpp models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="mixtral",
        choices=["mixtral", "mistral", "llama2-13b"],
        help="LLM model to use. L_4.",
    )
    parser.add_argument(
        "--root_dir",
        default="../Stage_1/",
        help="Root directory containing models and configs",
    )
    parser.add_argument(
        "-t",
        "--template",
        default="templates/generate_prompt.txt",
        help="Prompt template file",
    )
    parser.add_argument(
        "-c", "--concepts", required=True, help="Input concepts file (JSON/YAML/TXT)"
    )
    parser.add_argument("-d", "--domain", required=True, help="Target domain")
    parser.add_argument(
        "-cd", "--category-desc", required=True, help="Category description"
    )
    parser.add_argument(
        "-n", "--num-prompts", type=int, default=1, help="Number of prompts per concept"
    )
    parser.add_argument("-o", "--output", required=True, help="Output JSON file path")
    parser.add_argument(
        "--save_every", type=int, default=100, help="Save frequency (iterations)"
    )
    parser.add_argument(
        "--num_retries", type=int, default=10, help="Max retries per prompt"
    )

    args = parser.parse_args()
    main(args)
