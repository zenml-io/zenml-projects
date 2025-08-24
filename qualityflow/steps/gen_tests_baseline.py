"""
Generate baseline/skeleton tests using heuristics.
"""

import ast
import tempfile
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def gen_tests_baseline(
    workspace_dir: Path,
    code_summary: Dict,
    enabled: bool = True,
    max_files: int = 10,
) -> Annotated[Optional[Path], "baseline_tests_dir"]:
    """
    Generate baseline/skeleton tests using heuristic analysis.

    Args:
        workspace_dir: Path to workspace directory
        code_summary: Code analysis summary containing selected files
        enabled: Whether baseline generation is enabled
        max_files: Maximum number of files to process

    Returns:
        Path to baseline tests directory, or None if disabled
    """
    if not enabled:
        logger.info("Baseline test generation disabled")
        return None

    # Extract selected files from code summary
    selected_files = code_summary.get("selected_files", [])

    # Limit files if max_files is specified
    files_to_process = (
        selected_files[:max_files] if max_files > 0 else selected_files
    )
    logger.info(
        f"Generating baseline tests for {len(files_to_process)}/{len(selected_files)} files"
    )

    # Create baseline tests directory
    tests_dir = tempfile.mkdtemp(prefix="qualityflow_baseline_tests_")
    tests_path = Path(tests_dir)

    workspace_path = Path(workspace_dir)

    for file_path in files_to_process:
        logger.info(f"Generating baseline tests for {file_path}")

        # Read and parse source file
        full_file_path = workspace_path / file_path
        with open(full_file_path, "r") as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)

            # Extract functions and classes
            functions, classes = _extract_testable_items(tree)

            # Generate skeleton tests
            test_content = _generate_skeleton_tests(
                file_path, functions, classes
            )

            # Save baseline tests
            test_file_name = f"test_{Path(file_path).stem}_baseline.py"
            test_file_path = tests_path / test_file_name

            with open(test_file_path, "w") as f:
                f.write(test_content)

            logger.info(f"Baseline tests saved to {test_file_path}")

        except SyntaxError as e:
            logger.warning(f"Skipping {file_path} due to syntax error: {e}")
            continue

    logger.info("Baseline test generation complete")

    # Return Path object - ZenML will automatically materialize the folder
    return Path(tests_dir)


def _extract_testable_items(tree: ast.AST) -> tuple[List[str], List[str]]:
    """Extract function and class names from AST."""
    functions = []
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Skip private functions (starting with _)
            if not node.name.startswith("_"):
                functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            # Skip private classes
            if not node.name.startswith("_"):
                classes.append(node.name)

    return functions, classes


def _generate_skeleton_tests(
    file_path: str, functions: List[str], classes: List[str]
) -> str:
    """Generate skeleton test content."""

    # Create imports section
    imports = f'''"""
Baseline/skeleton tests for {file_path}
Generated using heuristic analysis.
"""

import pytest
import unittest
from unittest.mock import Mock, patch
'''

    # Try to determine import path from file path
    module_path = file_path.replace("/", ".").replace(".py", "")
    if module_path.startswith("src."):
        module_path = module_path[4:]  # Remove 'src.' prefix

    if functions or classes:
        imports += (
            f"# from {module_path} import {', '.join(functions + classes)}\n\n"
        )
    else:
        imports += f"# from {module_path} import *\n\n"

    # Generate function tests
    function_tests = ""
    for func_name in functions:
        function_tests += f'''
def test_{func_name}_basic():
    """Basic test for {func_name}."""
    # TODO: Implement test for {func_name}
    pass

def test_{func_name}_error_cases():
    """Error case test for {func_name}.""" 
    # TODO: Test error conditions for {func_name}
    pass
'''

    # Generate class tests
    class_tests = ""
    for class_name in classes:
        class_tests += f'''
class Test{class_name}(unittest.TestCase):
    """Test suite for {class_name}."""
    
    def setUp(self):
        """Set up test fixtures."""
        # TODO: Initialize test fixtures
        pass
    
    def test_{class_name.lower()}_init(self):
        """Test {class_name} initialization."""
        # TODO: Test class initialization
        pass
        
    def test_{class_name.lower()}_methods(self):
        """Test {class_name} methods."""
        # TODO: Test class methods
        pass
'''

    # Add default test if no functions or classes found
    if not functions and not classes:
        default_test = '''
class TestModule(unittest.TestCase):
    """Default test suite for module."""
    
    def test_module_imports(self):
        """Test that module can be imported."""
        # TODO: Add import test
        pass
'''
        class_tests += default_test

    # Combine all parts
    test_content = imports + function_tests + class_tests

    # Add main block
    test_content += """
if __name__ == "__main__":
    unittest.main()
"""

    return test_content
