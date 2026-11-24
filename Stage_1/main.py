# Script used to discover concepts in a domain using LLMs

# Arguments:
#   -v, --verbose: Verbose output
#   --model_seeds: Model to be used for generating seeds. Default is mixtral. Possible values are mixtral, mistral, llama2-13b.
#   --model_expand: Model to be used for expanding seeds. Default is mixtral. Possible values are mixtral, mistral, llama2-13b.
#   --model_check: Model to be used for checking concepts. Default is llama2-13b. Possible values are mixtral, mistral, llama2-13b.
#   -c, --config (str): Path of the config yml file to use. Required.
#   --num_seeds: Number of concepts to request per Step-1 call. Required (no safe default exists
#     across domains: the released Birds dataset used 50 and Insects 100; the value for Food
#     was not recorded).
#   --seed: Seed to use for inference. Default is None, which means a random seed is used.
#   --output_dir: Output directory to save results to. Required.
#   --ignore-existing: Ignore existing results in output directory. Default is False.

import argparse
from argparse import Namespace
import os
import pickle
import random
import sys
import time
import yaml
import json
import logging
from tqdm import tqdm
from copy import deepcopy

from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_configs(root_dir):
    if not os.path.exists(root_dir):
        raise ValueError(f"Provided root_dir '{root_dir}' does not exist.")

    # Define paths for available models
    MIXTRAL_PATH = os.path.join(
        root_dir,
        "models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",  # Path for mixtral model
    )
    MISTRAL_PATH = os.path.join(
        root_dir,
        "models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",  # Path for mistral model
    )
    LLAMA2_13B_PATH = os.path.join(
        root_dir, "models/llama-2-13b-chat.Q5_K_M.gguf"  # Path for llama2-13b model
    )

    # Configuration for each model
    LLM_DEFAULT_CONFIGS = {
        "mixtral": {
            "model": MIXTRAL_PATH,  # Model path for mixtral
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "max_tokens": 2048,
        },
        "mistral": {
            "model": MISTRAL_PATH,  # Model path for mistral
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "max_tokens": 2048,
        },
        "llama2-13b": {
            "model": LLAMA2_13B_PATH,  # Model path for llama2-13b
            "n_gpu_layers": -1,
            "n_ctx": 2048,
            "max_tokens": 2048,
        },
    }

    return LLM_DEFAULT_CONFIGS


LLM_DEFAULT_CONFIGS = build_configs(".")

GRAMMAR_SEED_CREATION = "grammars/json_seeds.gbnf"  # Grammar for Step 1
GRAMMAR_CONCEPT_CHECK = "grammars/true_false_answer.gbnf"  # Grammar for Step 3
GRAMMAR_SEED_EXPANSION = "grammars/json_seed_exp.gbnf"  # Grammar for Step 2

PROMPT_SEED_CREATION_TEMPLATE = "templates/generate_seeds.txt"  # Prompt for Step 1
PROMPT_CONCEPT_CHECK_TEMPLATE = "templates/check_concept.txt"  # Prompt for Step 3
PROMPT_SEED_EXPANSION_TEMPLATE = "templates/expand_seeds.txt"  # Prompt for Step 2


class LLMBuilder:
    def __init__(self, model, seed, verbose=False):
        self.model = model  # Model name (mixtral, mistral, llama2-13b)
        self.seed = seed  # Seed for reproducibility
        self.verbose = verbose
        self.llm_config = deepcopy(
            LLM_DEFAULT_CONFIGS[model]
        )  # Configuration for the selected model

    def build(self, grammar, **kwargs):
        # Update config with any additional kwargs
        config = deepcopy(self.llm_config)
        config.update(kwargs)

        return LlamaCpp(
            model_path=config["model"],  # Model path based on selected model
            n_gpu_layers=config["n_gpu_layers"],
            n_ctx=config["n_ctx"],
            f16_kv=True,
            verbose=self.verbose,
            grammar_path=grammar,  # Grammar file for specific step
            n_batch=config["n_ctx"],
            max_tokens=config["max_tokens"],
            seed=self.seed,  # Seed for model inference
            streaming=False,
        )


def check_relative_change_stopping_criteria(
    old_len: int,
    new_len: int,
    relative_threshold: float,
    iteration: int,
    total_iterations: int,
) -> bool:
    """
    Check if the relative change meets the stopping criteria.

    Args:
        old_len (int): Number of elements before the iteration
        new_len (int): Number of elements after the iteration
        relative_threshold (float): Threshold for relative change
        iteration (int): Current iteration number
        total_iterations (int): Total number of iterations allowed

    Returns:
        bool: True if stopping criteria is met, False otherwise
    """
    if relative_threshold is None:
        return False
    # Avoid division by zero
    if old_len == 0:
        return False

    # Calculate relative change
    relative_change = (new_len - old_len) / old_len

    # Log information about the change
    logger.info(f"Checking relative threshold: {new_len} vs {old_len}")
    logger.info(f"Relative change: {relative_change:.2f}")

    # Check if relative change is below the threshold
    if relative_change < relative_threshold:
        logger.info(f"Stopping at iteration {iteration+1}/{total_iterations}")
        return True

    return False


# Step 1: Generate seeds (Initial Concept Generation)
def generate_seeds(
    llm,  # Corresponds to L_1 (Initial Concept Generator)
    domain,  # $ n_{\mathcal{D}} $ (Domain name)
    category_desc,  # $ d_{\mathcal{D}} $ (Domain description)
    num_seeds,  # Number of concepts requested per call
    num_iters=1,  # $ N $ (Number of iterations)
    relative_threshold=None,  # $ \lambda_1 $ (Relative threshold)
    validate_concepts=False,  # Flag to validate concepts during seeding
):
    """
    Generate initial seeds using L_1.

    Note:
    - `validate_concepts` is an optional early filter, not the paper's
      validation step. It runs the same check prompt with the *generating*
      model (L_1 here), to avoid spending expansion compute on concepts that
      are already implausible; the released Birds and Insects datasets enabled
      it for that reason. The validation the paper describes is the final pass
      with a separate model L_3 (`--model_check`), which always runs and is
      what produces `valid_concepts.json`.
    """
    _llm = llm
    # Instantiate LLM with grammar for Step 1
    llm = llm(GRAMMAR_SEED_CREATION)  # Using grammar for Step 1
    parser = JsonOutputParser()

    # Load prompt from file
    with open(PROMPT_SEED_CREATION_TEMPLATE, "r") as f:
        prompt = f.read()
    prompt_template = PromptTemplate(
        template=prompt, input_variables=["domain", "category_desc", "num_seeds"]
    )
    llm_chain = prompt_template | llm | parser

    seeds = dict()  # $ \mathcal{B}_0 $ (Initial Concept Bank)

    for i in range(num_iters):  # Iterate up to $ N $ times
        logger.info(f"Generating seeds: iteration {i+1}/{num_iters}")
        sys.stderr.flush()
        sys.stdout.flush()
        output = llm_chain.invoke(
            dict(domain=domain, category_desc=category_desc, num_seeds=num_seeds)
        )
        old_len = len(seeds)
        for s in output["seeds"]:
            seeds[s.casefold()] = s
        new_len = len(seeds)
        if old_len != 0:
            logger.info(f"New seeds: {new_len - old_len}")
            logger.info(
                f"Relatively new seeds: {(new_len - old_len) / old_len * 100:.2f}%"
            )
            if check_relative_change_stopping_criteria(
                old_len, new_len, relative_threshold, i, num_iters
            ):
                break
        sys.stderr.flush()
        sys.stdout.flush()

    if validate_concepts:  # Optional early filter, run with L_1 (see docstring)
        logger.info("Checking concepts...")
        check_concepts_output = check_concepts(
            _llm, domain, category_desc, list(seeds.values())
        )
        seeds = {
            c.casefold(): c for c, output in check_concepts_output.items() if output
        }
        logger.info(f"Valid seeds: {len(seeds)}")

    return dict(seeds=list(seeds.values()))  # Returns $ \mathcal{B}_0 $


# Step 2: Expand seeds (Concept Expansion)
def expand_seeds(
    llm,  # Corresponds to L_2 (Concept Expander)
    domain,  # $ n_{\mathcal{D}} $ (Domain name)
    category_desc,  # $ d_{\mathcal{D}} $ (Domain description)
    seeds,  # $ \mathcal{B}_0 $ (Initial Concept Bank)
    num_iters,  # $ N $ (Number of iterations)
    target_num_concepts,  # Target number of concepts to reach
    force_stop=False,  # Flag to force stopping
    validate_concepts=False,  # Flag to validate concepts during expansion
    relative_threshold=None,  # $ \lambda_2 $ (Relative threshold)
):
    """
    Expand seeds using L_2.

    Note:
    - `validate_concepts` is an optional early filter, not the paper's
      validation step. It runs the same check prompt with the *generating*
      model (L_2 here), to avoid spending further expansion compute on
      concepts that are already implausible; the released Birds and Insects
      datasets enabled it for that reason. The validation the paper describes is
      the final pass with a separate model L_3 (`--model_check`), which
      always runs and is what produces `valid_concepts.json`.
    """
    # Instantiate LLM with grammar for Step 2
    _llm = llm
    llm = llm(GRAMMAR_SEED_EXPANSION)  # Using grammar for Step 2
    parser = JsonOutputParser()

    # Load prompt from file
    with open(PROMPT_SEED_EXPANSION_TEMPLATE, "r") as f:
        prompt = f.read()
    prompt_template = PromptTemplate(
        template=prompt, input_variables=["domain", "category_desc", "answer_1", "seed"]
    )

    llm_chain = prompt_template | llm | parser

    list_of_seeds = seeds["seeds"]  # List of seeds from $ \mathcal{B}_0 $
    concepts = dict()
    for s in list_of_seeds:
        concepts[s.casefold()] = s  # Initialize concepts with seeds

    if target_num_concepts is not None:
        pbar = tqdm(total=target_num_concepts, desc="Expanding seeds")
    else:
        pbar = tqdm(desc="Expanding seeds")

    # Update progress bar to reflect the number of concepts already discovered
    pbar.n = len(concepts)
    pbar.refresh()

    for i in range(num_iters):  # Iterate up to $ N $ times
        logger.info(f"Expanding seeds: iteration {i+1}/{num_iters}")
        if (target_num_concepts is not None) and (len(concepts) >= target_num_concepts):
            break
        init_concepts = set(concepts.values())  # Current concept bank $ \mathcal{B}_i $
        init_concepts_list = list(init_concepts)  # random.sample needs a sequence, not a set

        for seed in init_concepts:  # Expand each concept in $ \mathcal{B}_i $
            # Limit the size of the answer1 to 20 concepts to reduce the number of tokens
            sampled_seeds = set(
                random.sample(
                    init_concepts_list, min(20, len(init_concepts_list))
                )  # Sample up to 20 concepts
            )
            sampled_seeds.add(seed)

            answer_1 = str(list(sampled_seeds)).replace(
                "'", '"'
            )  # Prepare input for expansion

            try:
                output = llm_chain.invoke(
                    dict(
                        domain=domain,
                        category_desc=category_desc,
                        answer_1=answer_1,
                        seed=seed,
                    )
                )
                new_concepts = output["new_concepts"]
            except Exception as e:
                logger.error(f"Error while expanding seed {seed}: {e}")
                new_concepts = []

            for c in new_concepts:
                concepts[c.casefold()] = c  # Add new concepts to $ \mathcal{B}_{i+1} $

            pbar.n = len(concepts)
            pbar.refresh()

            if force_stop:
                if len(concepts) >= target_num_concepts:
                    break

        if validate_concepts:  # Optional early filter, run with L_2 (see docstring)
            new_concepts = list(concepts.values())

            check_concepts_output = check_concepts(
                _llm, domain, category_desc, new_concepts
            )
            old_len_new_concepts = len(new_concepts)
            new_concepts = [c for c, output in check_concepts_output.items() if output]
            new_len_new_concepts = len(new_concepts)
            logger.info(
                f"Valid concepts: {new_len_new_concepts}/{old_len_new_concepts}"
            )
            logger.info(
                f"Invalid concepts: {old_len_new_concepts - new_len_new_concepts}/{old_len_new_concepts}"
            )
            concepts = {c.casefold(): c for c in new_concepts}

        if check_relative_change_stopping_criteria(
            len(init_concepts), len(concepts), relative_threshold, i, num_iters
        ):
            break

    pbar.close()

    concepts = list(concepts.values())  # Final expanded concepts $ \mathcal{B}_N $

    return sorted(concepts)  # Returns $ \mathcal{B}_N $


# Step 3: Check concepts (Concept Filtering)
def check_concepts(
    llm,  # Corresponds to L_3 (Concept Validator)
    domain,  # $ n_{\mathcal{D}} $ (Domain name)
    category_desc,  # $ d_{\mathcal{D}} $ (Domain description)
    concepts,
):  # $ \mathcal{B}_N $ (Final Concept Bank)

    # Instantiate LLM with grammar for Step 3
    llm = llm(GRAMMAR_CONCEPT_CHECK)  # Using grammar for Step 3
    parser = JsonOutputParser()

    # Load prompt from file
    with open(PROMPT_CONCEPT_CHECK_TEMPLATE, "r") as f:
        prompt = f.read()
    prompt_template = PromptTemplate(
        template=prompt, input_variables=["domain", "category_desc", "concept"]
    )
    llm_chain = prompt_template | llm | parser

    outputs = {}  # Validation results

    pbar = tqdm(total=len(concepts), desc="Checking concepts")

    for concept in concepts:  # Validate each concept
        try:
            output = llm_chain.invoke(
                dict(domain=domain, category_desc=category_desc, concept=concept)
            )
        except Exception as e:
            logger.error(f"Error while checking concept {concept}: {e}")
            output = False
        outputs[concept] = output  # True if valid, else False
        pbar.update(1)
    pbar.close()

    return outputs  # Returns validation results


def process_and_save_output(func, args, output_filename, func_args, format="json"):
    """
    Process and save the output of a given function.

    :param func: The function to be executed.
    :param args: Command-line arguments.
    :param output_filename: The name of the file to save the output.
    :param func_args: Additional arguments required by the function.
    :param format: The format of the output. Default is json. Possible values are json, txt, yaml and pickle.
    """
    # Check if output directory is specified
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, output_filename)

        if os.path.exists(output_file) and not args.ignore_existing:
            raise ValueError(
                f"File {output_file} already exists. Use --ignore-existing to ignore existing results."
            )
        elif os.path.exists(output_file) and args.ignore_existing:
            logger.info(f"File {output_file} already exists. Skipping...")
            if format == "json":
                with open(output_file, "r") as f:
                    result = json.load(f)
            elif format == "txt":
                with open(output_file, "r") as f:
                    result = f.read()
            elif format == "yaml":
                with open(output_file, "r") as f:
                    result = yaml.load(f, Loader=yaml.SafeLoader)
            elif format == "pickle":
                with open(output_file, "rb") as f:
                    result = pickle.load(f)
        else:
            logger.info(
                f"Generating {output_filename[:-5]}..."  # Removing '.json' from filename for display
            )
            if args.verbose:
                start = time.time()
            result = func(**func_args)
            if args.verbose:
                stop = time.time()
                logger.info(f"Time elapsed: {stop-start}")
                logger.info(f"{output_filename[:-5].capitalize()}:")
                logger.info(result)

            # If output directory is provided, save result to specified format
            if format == "json":
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=4)
            elif format == "txt":
                with open(output_file, "w") as f:
                    f.write(result)
            elif format == "yaml":
                with open(output_file, "w") as f:
                    yaml.dump(result, f)
            elif format == "pickle":
                with open(output_file, "wb") as f:
                    pickle.dump(result, f)
    else:
        # If no output directory, just run the function and print the result
        result = func(**func_args)
        logger.info(result)

    return result


def main(args):
    # Print arguments
    logger.info(args)

    # Manage output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save arguments to pickle file
        args_file = os.path.join(args.output_dir, "args.pkl")
        if os.path.exists(args_file) and not args.ignore_existing:
            raise ValueError(
                f"File {args_file} already exists. Use --ignore-existing to ignore existing results."
            )
        elif os.path.exists(args_file) and args.ignore_existing:
            logger.info(
                f"File {args_file} already exists. Checking if arguments match..."
            )
            with open(args_file, "rb") as f:
                args_old = pickle.load(f)
            # Check if arguments match with existing arguments. If values are different, raise an error. If the new arguments have additional arguments, ignore them.
            # The value of --ignore-existing is ignored here
            if not all(
                [
                    getattr(args, arg) == getattr(args_old, arg)
                    for arg in vars(args_old)
                    if arg != "ignore_existing"
                ]
            ):
                raise ValueError(
                    f"Arguments do not match. Old arguments: {args_old}. New arguments: {args}."
                )
            else:
                logger.info(f"Arguments match. Skipping...")

            with open(args_file, "wb") as f:
                pickle.dump(args, f)
        else:
            with open(args_file, "wb") as f:
                pickle.dump(args, f)

    # Check if seed is provided. If not, generate a random seed
    if args.seed is None:
        args.seed = random.randint(
            0, 2**16 - 1
        )  # Generates a random seed if not provided
        logger.info(f"Random seed: {args.seed}")

    # The paper requires the validator L_3 to differ from the generators L_1
    # and L_2, so a concept is never judged by the model that proposed it.
    if args.model_check in (args.model_seeds, args.model_expand):
        logger.warning(
            f"--model_check ({args.model_check}) is the same model as "
            f"--model_seeds ({args.model_seeds}) or --model_expand "
            f"({args.model_expand}). The paper uses a validator L_3 distinct "
            f"from L_1 and L_2."
        )

    # Initialize LLM builder for generating seeds
    llm_builder_seeds = LLMBuilder(args.model_seeds, args.seed, args.verbose)  # L_1

    # First step: generate seeds
    logger.info("##### Generating seeds:")
    seeds_args = {
        "llm": llm_builder_seeds.build,  # Corresponds to L_1 (Initial Concept Generator)
        "domain": args.domain,  # $ n_{\mathcal{D}} $ (Domain name)
        "category_desc": args.category_desc,  # $ d_{\mathcal{D}} $ (Domain description)
        "num_seeds": args.num_seeds,  # Number of concepts requested per call
        "num_iters": args.num_iters_seeds,  # $ N $ (Number of iterations for Step 1)
        "relative_threshold": getattr(args, "relative_threshold_seeds", None),  # $ \lambda_1 $
        "validate_concepts": getattr(args, "validate_seeds", False),  # Optional early filter, run with L_1
    }
    seeds = process_and_save_output(
        generate_seeds, args, "seeds.json", seeds_args
    )  # Output $ \mathcal{B}_0 $
    sys.stderr.flush()
    sys.stdout.flush()

    # Initialize LLM builder for expanding seeds
    llm_builder_expand = LLMBuilder(args.model_expand, args.seed, args.verbose)  # L_2

    # Second step: Expand seeds
    logger.info("##### Expanding seeds:")
    expand_seeds_args = {
        "llm": llm_builder_expand.build,  # Corresponds to L_2 (Concept Expander)
        "domain": args.domain,  # $ n_{\mathcal{D}} $ (Domain name)
        "category_desc": args.category_desc,  # $ d_{\mathcal{D}} $ (Domain description)
        "seeds": seeds,  # $ \mathcal{B}_0 $ (Initial Concept Bank)
        "num_iters": args.seed_expansion[
            "num_iters"
        ],  # $ N $ (Number of iterations for Step 2)
        "target_num_concepts": args.seed_expansion[
            "target_num_concepts"
        ],  # Target number for expansion
        "force_stop": args.seed_expansion["force_stop"],  # Force stop flag
        "validate_concepts": args.seed_expansion.get(
            "validate_concepts", False
        ),  # Optional early filter, run with L_2
        "relative_threshold": args.seed_expansion.get(
            "relative_threshold", None
        ),  # $ \lambda_2 $
    }
    concepts = process_and_save_output(
        expand_seeds, args, "concepts.json", expand_seeds_args
    )  # Output $ \mathcal{B}_N $
    logger.info(f"Number of concepts: {len(concepts)}")
    sys.stderr.flush()
    sys.stdout.flush()

    # Initialize LLM builder for checking concepts
    llm_builder_check = LLMBuilder(args.model_check, args.seed, args.verbose)  # L_3

    # Intermediate step: check concepts (Validation)
    logger.info("##### Checking concepts:")
    check_concepts_args = {
        "llm": lambda x: llm_builder_check.build(
            x, max_tokens=128, n_ctx=1024
        ),  # Corresponds to L_3 (Concept Validator)
        "domain": args.domain,  # $ n_{\mathcal{D}} $ (Domain name)
        "category_desc": args.category_desc,  # $ d_{\mathcal{D}} $ (Domain description)
        "concepts": concepts,  # $ \mathcal{B}_N $ (Final Concept Bank)
    }
    check_concepts_output = process_and_save_output(
        check_concepts, args, "check_concepts.json", check_concepts_args
    )
    # Filter out concepts that are not valid
    concepts = [
        concept for concept, output in check_concepts_output.items() if output
    ]  # Final $ \mathcal{B} $
    logger.info(f"Number of valid concepts: {len(concepts)}")
    # Save valid concepts to file
    valid_concepts_file = os.path.join(args.output_dir, "valid_concepts.json")
    with open(valid_concepts_file, "w") as f:
        json.dump(concepts, f, indent=4)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Script used to discover concepts in a domain using LLMs"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--model_seeds",
        default="mixtral",
        choices=["mixtral", "mistral", "llama2-13b"],
        help="Model to be used for generating seeds. L_1. Default is mixtral. Possible values are mixtral, mistral, llama2-13b.",
    )
    parser.add_argument(
        "--model_expand",
        default="mixtral",
        choices=["mixtral", "mistral", "llama2-13b"],
        help="Model to be used for expanding seeds. L_2. Default is mixtral. Possible values are mixtral, mistral, llama2-13b.",
    )
    parser.add_argument(
        "--model_check",
        default="llama2-13b",
        choices=["mixtral", "mistral", "llama2-13b"],
        help="Model to be used for checking concepts. L_3. Default is llama2-13b. Possible values are mixtral, mistral, llama2-13b.",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path of the config yml file to use",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        required=True,
        help="Number of concepts to request per Step-1 call (no safe default exists across "
        "domains: the released Birds dataset used 50 and Insects 100; the value for Food "
        "was not recorded).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to use for inference. Default is None, which means a random seed is used.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory to save results to.",
    )
    parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Ignore existing results in output directory. Default is False.",
    )
    args = parser.parse_args()
    # Load yaml config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = Namespace(**config)
    # Add config arguments to args

    args = Namespace(**vars(args), **vars(config))
    main(args)
