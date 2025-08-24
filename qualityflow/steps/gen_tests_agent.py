"""
Generate tests using LLM agent.
"""

import tempfile
from pathlib import Path
from typing import Annotated, Dict, List, Tuple
from jinja2 import Template

from zenml import step
from zenml.logger import get_logger
from zenml import log_metadata
from enum import Enum


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
    Annotated[str, "prompt_used"]
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
        Tuple of test directory and prompt used
    """
    # Extract selected files from code summary
    selected_files = code_summary.get("selected_files", [])
    
    # Limit files if max_files is specified
    files_to_process = selected_files[:max_files] if max_files > 0 else selected_files
    logger.info(f"Generating tests for {len(files_to_process)}/{len(selected_files)} files using {provider}:{model}")
    
    # Create tests directory
    tests_dir = tempfile.mkdtemp(prefix="qualityflow_agent_tests_")
    tests_path = Path(tests_dir)
    
    # Load prompt template
    workspace_path = Path(workspace_dir)
    prompt_file = workspace_path / prompt_path
    
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
    else:
        # Use default template if file doesn't exist
        prompt_template = _get_default_prompt_template()
        logger.info(f"Using default prompt template, {prompt_path} not found")
    
    template = Template(prompt_template)
    
    total_tokens_in = 0
    total_tokens_out = 0
    materialized_prompts = {}  # Store materialized prompts per file
    
    for file_path in files_to_process:
        logger.info(f"Generating tests for {file_path}")
        
        # Read source file
        full_file_path = workspace_path / file_path
        with open(full_file_path, 'r') as f:
            source_code = f.read()
        
        # Render prompt
        materialized_prompt = template.render(
            file_path=file_path,
            source_code=source_code,
            max_tests=max_tests_per_file,
            complexity_score=code_summary.get("complexity_scores", {}).get(file_path, 0)
        )
        
        # Store the materialized prompt for this file
        materialized_prompts[file_path] = materialized_prompt
        
        # Generate tests using provider
        if provider == GenerationProvider.FAKE:
            generated_tests, tokens = _generate_fake_tests(file_path, source_code, max_tests_per_file)
        elif provider == GenerationProvider.OPENAI:
            generated_tests, tokens = _generate_openai_tests(materialized_prompt, model)
        elif provider == GenerationProvider.ANTHROPIC:
            generated_tests, tokens = _generate_anthropic_tests(materialized_prompt, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        total_tokens_in += tokens.get("tokens_in", 0)
        total_tokens_out += tokens.get("tokens_out", 0)
        
        # Save generated tests
        test_file_name = f"test_{Path(file_path).stem}.py"
        test_file_path = tests_path / test_file_name
        
        with open(test_file_path, 'w') as f:
            f.write(generated_tests)
        
        logger.info(f"Generated tests saved to {test_file_path}")
    
    # Log comprehensive metadata including materialized prompts
    metadata = {
        "token_usage": {
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "cost_estimate": _estimate_cost(total_tokens_in, total_tokens_out, provider, model),
        },
        "config": {
            "provider": provider.value,
            "model": model,
            "prompt_template_path": prompt_path,
            "max_tests_per_file": max_tests_per_file,
            "files_processed": len(files_to_process),
        },
        "materialized_prompts": materialized_prompts,
        "prompt_template": prompt_template,
    }
    
    log_metadata(metadata)
    logger.info(f"Test generation complete. Files: {len(files_to_process)}, Tokens: {total_tokens_in} in / {total_tokens_out} out")
    
    # Create a better prompt summary for the report
    prompt_summary = f"Template: {prompt_path}\nProvider: {provider.value}\nModel: {model}\nFiles processed: {len(files_to_process)}"
    
    # Return Path object - ZenML will automatically materialize the folder
    return Path(tests_dir), prompt_summary


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


def _generate_fake_tests(file_path: str, source_code: str, max_tests: int) -> Tuple[str, Dict]:
    """Generate fake/mock tests for development/testing."""
    # Create a simple module name from file path
    module_name = file_path.replace('/', '.').replace('.py', '')
    
    test_content = f'''"""
Generated tests for {file_path}
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock

class Test{file_path.split('/')[-1].replace('.py', '').title()}(unittest.TestCase):
    """Auto-generated test class for {file_path}."""
    
    def test_module_import(self):
        """Test that we can at least validate the test framework."""
        # Simple test that always passes to ensure test discovery works
        self.assertTrue(True)
        
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Mock test demonstrating test execution
        result = 1 + 1
        self.assertEqual(result, 2)
        
    def test_error_handling(self):
        """Test error handling."""
        # Test exception handling
        with self.assertRaises(ValueError):
            raise ValueError("Expected test exception")
            
    def test_mock_usage(self):
        """Test mock functionality."""
        # Test using mocks
        mock_obj = Mock()
        mock_obj.method.return_value = "mocked_result"
        result = mock_obj.method()
        self.assertEqual(result, "mocked_result")
        
    def test_coverage_target(self):
        """Test that generates some coverage."""
        # Simple operations to generate coverage
        data = {{"key": "value"}}
        self.assertIn("key", data)
        
        items = [1, 2, 3, 4, 5]
        filtered = [x for x in items if x > 3]
        self.assertEqual(len(filtered), 2)

if __name__ == "__main__":
    unittest.main()
'''
    
    tokens = {"tokens_in": 100, "tokens_out": 50}
    return test_content, tokens


def _generate_openai_tests(prompt: str, model: str) -> Tuple[str, Dict]:
    """Generate tests using OpenAI API."""
    try:
        import openai
        import os
        
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
                {"role": "system", "content": "You are a Python test generation expert. Generate comprehensive unit tests for the given code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
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
            "tokens_out": response.usage.completion_tokens
        }
        
        logger.info(f"Generated tests using OpenAI {model}: {tokens['tokens_in']} in, {tokens['tokens_out']} out")
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
        import anthropic
        import os
        
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
                {"role": "user", "content": f"You are a Python test generation expert. Generate comprehensive unit tests for the given code.\n\n{prompt}"}
            ]
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
            "tokens_out": response.usage.output_tokens
        }
        
        logger.info(f"Generated tests using Anthropic {model}: {tokens['tokens_in']} in, {tokens['tokens_out']} out")
        return test_content, tokens
        
    except ImportError:
        logger.warning("Anthropic library not installed, using fake tests")
        return _generate_fake_tests("anthropic_file", "mock_code", 3)
    except Exception as e:
        logger.error(f"Failed to generate tests with Anthropic: {e}")
        logger.warning("Falling back to fake tests")
        return _generate_fake_tests("anthropic_file", "mock_code", 3)


def _estimate_cost(tokens_in: int, tokens_out: int, provider: GenerationProvider, model: str) -> float:
    """Estimate cost based on token usage."""
    # Rough cost estimates (would need real pricing)
    if provider == GenerationProvider.OPENAI:
        if "gpt-4" in model:
            return (tokens_in * 0.00003) + (tokens_out * 0.00006)
        else:  # gpt-3.5
            return (tokens_in * 0.0000015) + (tokens_out * 0.000002)
    elif provider == GenerationProvider.ANTHROPIC:
        return (tokens_in * 0.000008) + (tokens_out * 0.000024)
    else:
        return 0.0