#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.2.0"]
# ///
"""Regenerate curated HumanEval subset JSON files from HuggingFace.

Usage:
    uv run scripts/regenerate_humaneval_subset.py
    uv run scripts/regenerate_humaneval_subset.py --sample-size 20 --seed 123
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import click
from datasets import load_dataset

# Task IDs known to be challenging for LLMs
HARD_TASK_IDS = {
    "HumanEval/4",   # mean_absolute_deviation
    "HumanEval/32",  # polynomial find_zero
    "HumanEval/95",  # check_dict_case
    "HumanEval/161", # solve (reverse case)
}


@click.command()
@click.option("--sample-size", default=12, help="Non-hard problems to sample")
@click.option("--seed", default=42, help="Random seed")
@click.option(
    "--output",
    default="test_cases/humaneval_subset.json",
    help="Output path for main subset",
)
@click.option(
    "--hard-output",
    default="test_cases/hard_problems.json",
    help="Output path for hard problems",
)
def main(
    sample_size: int, seed: int, output: str, hard_output: str
) -> None:
    """Download HumanEval and create curated subset files."""
    project_root = Path(__file__).parent.parent

    print("Loading HumanEval dataset from HuggingFace...")
    ds = load_dataset("openai/openai_humaneval", split="test")

    # Separate hard vs non-hard
    hard_problems = []
    other_problems = []

    for row in ds:
        record = {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "canonical_solution": row["canonical_solution"],
            "entry_point": row["entry_point"],
            "difficulty": (
                "hard" if row["task_id"] in HARD_TASK_IDS else "medium"
            ),
        }
        if row["task_id"] in HARD_TASK_IDS:
            hard_problems.append(record)
        else:
            other_problems.append(record)

    # Sample non-hard problems deterministically
    rng = random.Random(seed)
    rng.shuffle(other_problems)

    # Mark a few as "easy" (first 3 by sorted task_id)
    other_problems.sort(key=lambda x: x["task_id"])
    for i, p in enumerate(other_problems):
        if i < 3:
            p["difficulty"] = "easy"

    rng.shuffle(other_problems)
    sampled = other_problems[:sample_size]

    # Write files
    subset_path = project_root / output
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(subset_path, "w") as f:
        json.dump(sampled, f, indent=2)
    print(f"Wrote {len(sampled)} problems to {subset_path}")

    hard_path = project_root / hard_output
    hard_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hard_path, "w") as f:
        json.dump(hard_problems, f, indent=2)
    print(f"Wrote {len(hard_problems)} hard problems to {hard_path}")


if __name__ == "__main__":
    main()
