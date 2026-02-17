"""Step: Load and curate HumanEval test cases.

Loads a pre-curated JSON subset of HumanEval problems, ensures
hard problems are included, and samples deterministically.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Annotated

from zenml import log_metadata, step

from utils.scoring import TestCase

logger = logging.getLogger(__name__)


@step
def load_test_cases(
    subset_path: str = "test_cases/humaneval_subset.json",
    hard_problems_path: str = "test_cases/hard_problems.json",
    sample_size: int = 15,
    seed: int = 42,
    min_hard: int = 3,
) -> Annotated[list[TestCase], "test_cases"]:
    """Load curated HumanEval subset with guaranteed hard problem inclusion.

    Strategy: load the main subset + hard problems, ensure at least
    `min_hard` hard problems are in the final sample, then fill
    remaining slots from medium/easy problems.
    """
    project_root = Path(__file__).parent.parent

    # Load main subset
    subset_file = project_root / subset_path
    with open(subset_file) as f:
        subset_raw = json.load(f)
    subset = [TestCase(**tc) for tc in subset_raw]

    # Load hard problems
    hard_file = project_root / hard_problems_path
    with open(hard_file) as f:
        hard_raw = json.load(f)
    hard_cases = [TestCase(**tc) for tc in hard_raw]

    # Merge: add hard problems not already in subset
    subset_ids = {tc.task_id for tc in subset}
    for hc in hard_cases:
        if hc.task_id not in subset_ids:
            subset.append(hc)
            subset_ids.add(hc.task_id)

    # Separate hard vs non-hard
    hard = [tc for tc in subset if tc.difficulty == "hard"]
    non_hard = [tc for tc in subset if tc.difficulty != "hard"]

    # Guarantee minimum hard problems
    rng = random.Random(seed)
    selected_hard = hard[: min(min_hard, len(hard))]

    # Fill remaining slots from non-hard
    remaining = sample_size - len(selected_hard)
    rng.shuffle(non_hard)
    selected_non_hard = non_hard[:remaining]

    final = selected_hard + selected_non_hard
    rng.shuffle(final)

    logger.info(
        "Loaded %d test cases (%d hard, %d other)",
        len(final),
        len(selected_hard),
        len(selected_non_hard),
    )

    log_metadata(
        metadata={
            "total_test_cases": len(final),
            "hard_count": len(selected_hard),
            "task_ids": [tc.task_id for tc in final],
            "seed": seed,
        }
    )

    return final
