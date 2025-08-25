"""
Analyze and select code files for test generation.
"""

import ast
import glob
import os
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, List

from zenml import step
from zenml.logger import get_logger


class SelectionStrategy(str, Enum):
    """Code file selection strategies."""

    LOW_COVERAGE = "low_coverage"
    CHANGED_FILES = "changed_files"
    ALL = "all"


logger = get_logger(__name__)


@step
def analyze_code(
    workspace_dir: Path,
    commit_sha: str,
    source_spec: Dict[str, str],
    strategy: SelectionStrategy = SelectionStrategy.LOW_COVERAGE,
    max_files: int = 10,
) -> Annotated[Dict, "code_summary"]:
    """
    Analyze workspace and select candidate files for test generation.

    Args:
        workspace_dir: Path to workspace directory
        commit_sha: Git commit SHA
        source_spec: Source specification containing target_glob and other settings
        strategy: File selection strategy
        max_files: Maximum number of files to select

    Returns:
        Code summary dictionary containing selected files and metadata
    """
    # Extract target_glob from source spec
    target_glob = source_spec.get("target_glob", "src/**/*.py")

    logger.info(
        f"Analyzing code in {workspace_dir} with strategy {strategy} and glob {target_glob}"
    )

    workspace_path = Path(workspace_dir)

    # Find all Python files matching glob pattern
    all_files = []
    for pattern in target_glob.split(","):
        pattern = pattern.strip()
        matched_files = glob.glob(
            str(workspace_path / pattern), recursive=True
        )
        all_files.extend(matched_files)

    # Make paths relative to workspace
    relative_files = [
        os.path.relpath(f, workspace_dir)
        for f in all_files
        if f.endswith(".py") and os.path.isfile(f)
    ]

    logger.info(f"Found {len(relative_files)} Python files")

    # Calculate complexity scores
    complexity_scores = {}
    valid_files = []

    for file_path in relative_files:
        full_path = workspace_path / file_path
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST and calculate basic complexity
            tree = ast.parse(content)
            complexity = _calculate_complexity(tree)
            complexity_scores[file_path] = complexity
            valid_files.append(file_path)

        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Skipping {file_path} due to parsing error: {e}")
            continue

    # Select files based on strategy
    selected_files = _select_files(
        valid_files, complexity_scores, strategy, max_files
    )

    code_summary = {
        "selected_files": selected_files,
        "total_files": len(valid_files),
        "selection_reason": f"Selected top {len(selected_files)} files using {strategy} strategy",
        "complexity_scores": {f: complexity_scores[f] for f in selected_files},
    }

    logger.info(f"Selected {len(selected_files)} files: {selected_files}")

    return code_summary


def _calculate_complexity(tree: ast.AST) -> float:
    """Calculate basic complexity score for an AST."""

    class ComplexityVisitor(ast.NodeVisitor):
        def __init__(self):
            self.complexity = 0
            self.functions = 0
            self.classes = 0

        def visit_FunctionDef(self, node):
            self.functions += 1
            self.complexity += 1
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                    self.complexity += 1
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            self.classes += 1
            self.complexity += 1
            self.generic_visit(node)

    visitor = ComplexityVisitor()
    visitor.visit(tree)

    # Combine metrics into single score
    return visitor.complexity + visitor.functions * 0.5 + visitor.classes * 2


def _select_files(
    files: List[str],
    complexity_scores: Dict[str, float],
    strategy: SelectionStrategy,
    max_files: int,
) -> List[str]:
    """Select files based on strategy."""

    if strategy == SelectionStrategy.ALL:
        return files[:max_files]

    elif strategy == SelectionStrategy.LOW_COVERAGE:
        # Prioritize complex files that likely need more tests
        sorted_files = sorted(
            files, key=lambda f: complexity_scores[f], reverse=True
        )
        return sorted_files[:max_files]

    elif strategy == SelectionStrategy.CHANGED_FILES:
        # For this demo, just return all files (in real implementation, would use git diff)
        logger.warning(
            "CHANGED_FILES strategy not fully implemented, falling back to ALL"
        )
        return files[:max_files]

    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")
