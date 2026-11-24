"""Shared prompt-list construction for Stage 2's synthetic images.

`scripts/main_synth.py` (which renders the images) and
`scripts/generate_wds_from_synth.py` (which packs them into a webdataset)
must build the identical per-concept prompt list: the file names the
generator writes are derived from each prompt's index, and the converter
looks images up by exactly that name.
"""

import json
from typing import Dict, List


def load_concept_prompts(prompts_path: str) -> Dict[str, List[str]]:
    """Load a prompts JSON file into a {concept: [prompt, ...]} mapping.

    Args:
        prompts_path: Path to the JSON file written by `scripts/main_prompts.py`.

    Returns:
        Each concept mapped to its ordered list of generated prompts.
    """
    with open(prompts_path, "r") as f:
        prompts = json.load(f)

    return {
        concept: [
            d["image_caption"]
            for d in entries
            if "image_caption" in d and d["image_caption"].strip()
        ]
        for concept, entries in prompts.items()
    }
