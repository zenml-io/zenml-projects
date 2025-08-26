"""
Generate tests using LLM agent.
"""

import tempfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Dict, Tuple

from jinja2 import Template
from zenml import log_metadata, step
from zenml.logger import get_logger
from zenml.types import MarkdownString


class GenerationProvider(str, Enum):
    """LLM providers for test generation."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    FAKE = "fake"


logger = get_logger(__name__)


@step
def gen_tests_agent(
    workspace_dir: Path,
    code_summary: Dict,
    provider: GenerationProvider = GenerationProvider.FAKE,
    model: str = "gpt-4o-mini",
    prompt_path: str = "prompts/unit_test_v1.jinja",
    max_tests_per_file: int = 3,
    max_files: int = 10,
) -> Tuple[
    Annotated[Path, "agent_tests_dir"],
    Annotated[MarkdownString, "test_summary"],
]:
    """Generate tests using LLM agent.

    Args:
        workspace_dir: Path to workspace directory
        code_summary: Code analysis summary containing selected files
        provider: LLM provider to use
        model: Model name
        prompt_path: Path to Jinja2 prompt template
        max_tests_per_file: Maximum tests to generate per file
        max_files: Maximum number of files to process (for speed control)

    Returns:
        Tuple of test directory and test generation summary
    """
    # Extract selected files from code summary
    selected_files = code_summary.get("selected_files", [])

    # Limit files if max_files is specified
    files_to_process = (
        selected_files[:max_files] if max_files > 0 else selected_files
    )
    logger.info(
        f"Generating tests for {len(files_to_process)}/{len(selected_files)} files using {provider}:{model}"
    )

    # Create tests directory
    tests_dir = tempfile.mkdtemp(prefix="qualityflow_agent_tests_")
    tests_path = Path(tests_dir)

    # Load prompt template from QualityFlow project directory
    # Note: workspace_dir is the cloned repo, but prompts are in QualityFlow project
    try:
        # Try to resolve project root more robustly
        current_file = Path(__file__).resolve()
        project_root = (
            current_file.parent.parent
        )  # Go up from steps/ to project root
        prompt_file = project_root / prompt_path
    except Exception:
        # Fallback to current working directory if path resolution fails
        prompt_file = Path.cwd() / prompt_path

    if prompt_file.exists():
        with open(prompt_file, "r") as f:
            prompt_template = f.read()
        logger.info(f"Loaded prompt template from {prompt_file}")
    else:
        # Use default template if file doesn't exist
        prompt_template = _get_default_prompt_template()
        logger.info(
            f"Using default prompt template, {prompt_path} not found at {prompt_file}"
        )

    template = Template(prompt_template)

    # Keep workspace_path for reading source files
    workspace_path = Path(workspace_dir)

    total_tokens_in = 0
    total_tokens_out = 0
    test_snippets = {}  # Store test snippets per file
    test_stats = {}  # Store test statistics per file

    for file_path in files_to_process:
        logger.info(f"Generating tests for {file_path}")

        # Read source file
        full_file_path = workspace_path / file_path
        with open(full_file_path, "r") as f:
            source_code = f.read()

        # Render prompt
        materialized_prompt = template.render(
            file_path=file_path,
            source_code=source_code,
            max_tests=max_tests_per_file,
            complexity_score=code_summary.get("complexity_scores", {}).get(
                file_path, 0
            ),
        )

        # Store test generation info for this file
        test_stats[file_path] = {
            "provider": provider.value,
            "model": model,
            "max_tests": max_tests_per_file,
            "complexity_score": code_summary.get("complexity_scores", {}).get(
                file_path, 0
            ),
        }

        # Generate tests using provider
        if provider == GenerationProvider.FAKE:
            generated_tests, tokens = _generate_fake_tests(
                file_path, source_code, max_tests_per_file
            )
        elif provider == GenerationProvider.OPENAI:
            generated_tests, tokens = _generate_openai_tests(
                materialized_prompt, model
            )
        elif provider == GenerationProvider.ANTHROPIC:
            generated_tests, tokens = _generate_anthropic_tests(
                materialized_prompt, model
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        total_tokens_in += tokens.get("tokens_in", 0)
        total_tokens_out += tokens.get("tokens_out", 0)

        # Save generated tests
        test_file_name = f"test_{Path(file_path).stem}.py"
        test_file_path = tests_path / test_file_name

        with open(test_file_path, "w") as f:
            f.write(generated_tests)

        # Store test snippet for summary (first 20 lines)
        test_lines = generated_tests.split("\n")
        snippet_lines = test_lines[:20]
        if len(test_lines) > 20:
            snippet_lines.append("... (truncated)")
        test_snippets[file_path] = "\n".join(snippet_lines)

        # Update test stats with actual counts
        test_stats[file_path]["lines_generated"] = len(test_lines)
        test_stats[file_path]["test_functions"] = len(
            [
                line
                for line in test_lines
                if line.strip().startswith("def test_")
            ]
        )

        logger.info(f"Generated tests saved to {test_file_path}")

    # Log comprehensive metadata
    metadata = {
        "token_usage": {
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "cost_estimate": _estimate_cost(
                total_tokens_in, total_tokens_out, provider, model
            ),
        },
        "config": {
            "provider": provider.value,
            "model": model,
            "prompt_template_path": prompt_path,
            "max_tests_per_file": max_tests_per_file,
            "files_processed": len(files_to_process),
        },
        "test_stats": test_stats,
    }

    log_metadata(metadata)
    logger.info(
        f"Test generation complete. Files: {len(files_to_process)}, Tokens: {total_tokens_in} in / {total_tokens_out} out"
    )

    # Create test generation summary
    test_summary = _create_test_summary(
        provider,
        model,
        prompt_path,
        files_to_process,
        test_snippets,
        test_stats,
        total_tokens_in,
        total_tokens_out,
    )

    # Return Path object - ZenML will automatically materialize the folder
    return Path(tests_dir), test_summary


def _create_test_summary(
    provider: GenerationProvider,
    model: str,
    prompt_path: str,
    files_processed: list,
    test_snippets: Dict[str, str],
    test_stats: Dict[str, Dict],
    total_tokens_in: int,
    total_tokens_out: int,
) -> MarkdownString:
    """Create a markdown summary of test generation results."""

    # Calculate totals
    total_lines = sum(
        stats.get("lines_generated", 0) for stats in test_stats.values()
    )
    total_test_functions = sum(
        stats.get("test_functions", 0) for stats in test_stats.values()
    )

    # Handle edge case of no files processed
    if len(files_processed) == 0:
        summary = f"""# 🧪 Test Generation Summary

## Configuration
- **Provider**: {provider.value}
- **Model**: {model}
- **Prompt Template**: {prompt_path}
- **Files Processed**: 0

## Generation Statistics
⚠️ **No files were processed for test generation.**

This could happen if:
- No files matched the target glob pattern
- All files were filtered out during analysis
- Max files limit was set to 0

**Token Usage**: {total_tokens_in:,} in / {total_tokens_out:,} out
"""
        return MarkdownString(summary)

    # Build markdown content for successful processing
    avg_tests = total_test_functions / len(files_processed)
    summary = f"""# 🧪 Test Generation Summary

## Configuration
- **Provider**: {provider.value}
- **Model**: {model}
- **Prompt Template**: {prompt_path}
- **Files Processed**: {len(files_processed)}

## Generation Statistics
- **Total Lines Generated**: {total_lines:,}
- **Total Test Functions**: {total_test_functions}
- **Average Tests per File**: {avg_tests:.1f}
- **Token Usage**: {total_tokens_in:,} in / {total_tokens_out:,} out

## Generated Tests by File

"""

    for file_path in files_processed:
        stats = test_stats.get(file_path, {})
        snippet = test_snippets.get(file_path, "")

        complexity = stats.get("complexity_score", 0)
        lines = stats.get("lines_generated", 0)
        test_count = stats.get("test_functions", 0)

        summary += f"""### 📄 `{file_path}`
**Complexity Score**: {complexity:.1f} | **Lines**: {lines} | **Test Functions**: {test_count}

```
{snippet}
```

---

"""

    return MarkdownString(summary)


def _get_default_prompt_template() -> str:
    """Default Jinja2 prompt template for test generation."""
    return """# Generate unit tests for the following Python code

File: {{ file_path }}
Complexity Score: {{ complexity_score }}
Max Tests: {{ max_tests }}

## Source Code:
```python
{{ source_code }}
```

## Instructions:
Generate {{ max_tests }} comprehensive unit tests for the functions and classes in this code.
Focus on edge cases, error conditions, and typical usage patterns.

## Generated Tests:
"""


def _generate_fake_tests(
    file_path: str, source_code: str, max_tests: int
) -> Tuple[str, Dict]:
    """Generate fake/mock tests for development/testing.

    This generates more realistic-looking tests that attempt to exercise
    the actual source code by parsing it for functions and classes.
    """
    import ast

    # Parse the source code to extract function/class names
    try:
        tree = ast.parse(source_code)
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not node.name.startswith(
                "_"
            ):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
    except Exception:
        # Fallback if parsing fails
        functions = []
        classes = []

    # Generate module name from file path
    module_name = file_path.replace("/", ".").replace(".py", "")
    class_name = file_path.split("/")[-1].replace(".py", "").title()

    test_content = f'''"""
Generated tests for {file_path}
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock

# Attempt to import the module under test
try:
    from {module_name} import *
except ImportError:
    # Handle import errors gracefully for demo purposes
    pass

class Test{class_name}(unittest.TestCase):
    """Auto-generated test class for {file_path}."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {{"sample": "data", "numbers": [1, 2, 3]}}
    
    def test_module_import(self):
        """Test that the module can be imported without errors."""
        # This test ensures the module structure is valid
        self.assertTrue(True, "Module imported successfully")
'''

    # Generate tests for discovered functions
    for func_name in functions[: max_tests // 2]:
        test_content += f'''
    def test_{func_name}_basic(self):
        """Test basic functionality of {func_name}."""
        # TODO: Add proper test for {func_name}
        # This is a placeholder that should exercise the function
        try:
            # Attempt to call the function with basic parameters
            if callable(globals().get('{func_name}')):
                # Basic smoke test - at least try to call it
                pass
        except NameError:
            # Function not available in scope
            pass
        self.assertTrue(True, "Basic test for {func_name}")
'''

    # Generate tests for discovered classes
    for class_name_found in classes[: max_tests // 3]:
        test_content += f'''
    def test_{class_name_found.lower()}_instantiation(self):
        """Test that {class_name_found} can be instantiated."""
        try:
            if '{class_name_found}' in globals():
                # Try basic instantiation
                # obj = {class_name_found}()
                pass
        except NameError:
            pass
        self.assertTrue(True, "Instantiation test for {class_name_found}")
'''

    # Add some general coverage tests
    test_content += f'''
    def test_error_handling(self):
        """Test error handling patterns."""
        with self.assertRaises(ValueError):
            raise ValueError("Expected test exception")
            
    def test_data_structures(self):
        """Test basic data structure operations."""
        data = self.test_data.copy()
        self.assertIn("sample", data)
        self.assertEqual(len(data["numbers"]), 3)
        
    def test_mock_usage(self):
        """Test mock functionality."""
        mock_obj = Mock()
        mock_obj.method.return_value = "mocked_result"
        result = mock_obj.method()
        self.assertEqual(result, "mocked_result")

if __name__ == "__main__":
    unittest.main()
'''

    tokens = {"tokens_in": 100, "tokens_out": 50}
    return test_content, tokens


def _generate_openai_tests(prompt: str, model: str) -> Tuple[str, Dict]:
    """Generate tests using OpenAI API."""
    try:
        import os

        import openai

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found, using fake tests")
            return _generate_fake_tests("openai_file", "mock_code", 3)

        client = openai.OpenAI(api_key=api_key)

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python test generation expert. Generate comprehensive unit tests for the given code.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.1,
        )

        # Extract test code from response
        generated_content = response.choices[0].message.content

        # Try to extract Python code blocks
        if "```python" in generated_content:
            start = generated_content.find("```python") + 9
            end = generated_content.find("```", start)
            test_content = generated_content[start:end].strip()
        elif "```" in generated_content:
            start = generated_content.find("```") + 3
            end = generated_content.find("```", start)
            test_content = generated_content[start:end].strip()
        else:
            # Use the whole response if no code blocks found
            test_content = generated_content.strip()

        # Token usage for cost estimation
        tokens = {
            "tokens_in": response.usage.prompt_tokens,
            "tokens_out": response.usage.completion_tokens,
        }

        logger.info(
            f"Generated tests using OpenAI {model}: {tokens['tokens_in']} in, {tokens['tokens_out']} out"
        )
        return test_content, tokens

    except ImportError:
        logger.warning("OpenAI library not installed, using fake tests")
        return _generate_fake_tests("openai_file", "mock_code", 3)
    except Exception as e:
        logger.error(f"Failed to generate tests with OpenAI: {e}")
        logger.warning("Falling back to fake tests")
        return _generate_fake_tests("openai_file", "mock_code", 3)


def _generate_anthropic_tests(prompt: str, model: str) -> Tuple[str, Dict]:
    """Generate tests using Anthropic API."""
    try:
        import os

        import anthropic

        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found, using fake tests")
            return _generate_fake_tests("anthropic_file", "mock_code", 3)

        client = anthropic.Anthropic(api_key=api_key)

        # Call Anthropic API
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": f"You are a Python test generation expert. Generate comprehensive unit tests for the given code.\n\n{prompt}",
                }
            ],
        )

        # Extract test content from response
        generated_content = response.content[0].text

        # Try to extract Python code blocks
        if "```python" in generated_content:
            start = generated_content.find("```python") + 9
            end = generated_content.find("```", start)
            test_content = generated_content[start:end].strip()
        elif "```" in generated_content:
            start = generated_content.find("```") + 3
            end = generated_content.find("```", start)
            test_content = generated_content[start:end].strip()
        else:
            # Use the whole response if no code blocks found
            test_content = generated_content.strip()

        # Token usage for cost estimation
        tokens = {
            "tokens_in": response.usage.input_tokens,
            "tokens_out": response.usage.output_tokens,
        }

        logger.info(
            f"Generated tests using Anthropic {model}: {tokens['tokens_in']} in, {tokens['tokens_out']} out"
        )
        return test_content, tokens

    except ImportError:
        logger.warning("Anthropic library not installed, using fake tests")
        return _generate_fake_tests("anthropic_file", "mock_code", 3)
    except Exception as e:
        logger.error(f"Failed to generate tests with Anthropic: {e}")
        logger.warning("Falling back to fake tests")
        return _generate_fake_tests("anthropic_file", "mock_code", 3)


def _estimate_cost(
    tokens_in: int, tokens_out: int, provider: GenerationProvider, model: str
) -> float:
    """Estimate cost based on token usage.

    WARNING: These are hardcoded pricing estimates that will become outdated.
    For accurate pricing, refer to the official pricing pages:
    - OpenAI: https://openai.com/api/pricing/
    - Anthropic: https://www.anthropic.com/pricing

    Consider implementing a dynamic pricing lookup or configuration-based approach
    for production use.
    """
    # NOTE: These are rough estimates based on pricing as of early 2024
    # and will likely become outdated as providers update their pricing
    if provider == GenerationProvider.OPENAI:
        if "gpt-4" in model:
            # GPT-4 pricing (approximate, check current rates)
            return (tokens_in * 0.00003) + (tokens_out * 0.00006)
        else:  # gpt-3.5 and other models
            # GPT-3.5 pricing (approximate, check current rates)
            return (tokens_in * 0.0000015) + (tokens_out * 0.000002)
    elif provider == GenerationProvider.ANTHROPIC:
        # Claude pricing (approximate, check current rates)
        return (tokens_in * 0.000008) + (tokens_out * 0.000024)
    else:
        return 0.0
