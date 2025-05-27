"""Unit tests for prompt loader utilities."""

import pytest
from utils.prompt_loader import get_prompt_for_step, load_prompts_bundle
from utils.prompt_models import PromptsBundle


class TestPromptLoader:
    """Test cases for prompt loader functions."""

    def test_load_prompts_bundle(self):
        """Test loading prompts bundle from prompts.py."""
        bundle = load_prompts_bundle(pipeline_version="2.0.0")

        # Check it returns a PromptsBundle
        assert isinstance(bundle, PromptsBundle)

        # Check pipeline version
        assert bundle.pipeline_version == "2.0.0"

        # Check all core prompts are loaded
        assert bundle.search_query_prompt is not None
        assert bundle.query_decomposition_prompt is not None
        assert bundle.synthesis_prompt is not None
        assert bundle.viewpoint_analysis_prompt is not None
        assert bundle.reflection_prompt is not None
        assert bundle.additional_synthesis_prompt is not None
        assert bundle.conclusion_generation_prompt is not None

        # Check prompts have correct metadata
        assert bundle.search_query_prompt.name == "search_query_prompt"
        assert (
            bundle.search_query_prompt.description
            == "Generates effective search queries from sub-questions"
        )
        assert bundle.search_query_prompt.version == "1.0.0"
        assert "search" in bundle.search_query_prompt.tags

        # Check that actual prompt content is loaded
        assert "search query" in bundle.search_query_prompt.content.lower()
        assert "json schema" in bundle.search_query_prompt.content.lower()

    def test_load_prompts_bundle_default_version(self):
        """Test loading prompts bundle with default version."""
        bundle = load_prompts_bundle()
        assert bundle.pipeline_version == "1.0.0"

    def test_get_prompt_for_step(self):
        """Test getting prompt content for specific steps."""
        bundle = load_prompts_bundle()

        # Test valid step names
        test_cases = [
            ("query_decomposition", "query_decomposition_prompt"),
            ("search_query_generation", "search_query_prompt"),
            ("synthesis", "synthesis_prompt"),
            ("viewpoint_analysis", "viewpoint_analysis_prompt"),
            ("reflection", "reflection_prompt"),
            ("additional_synthesis", "additional_synthesis_prompt"),
            ("conclusion_generation", "conclusion_generation_prompt"),
        ]

        for step_name, expected_prompt_attr in test_cases:
            content = get_prompt_for_step(bundle, step_name)
            expected_content = getattr(bundle, expected_prompt_attr).content
            assert content == expected_content

    def test_get_prompt_for_step_invalid(self):
        """Test getting prompt for invalid step name."""
        bundle = load_prompts_bundle()

        with pytest.raises(
            ValueError, match="No prompt mapping found for step: invalid_step"
        ):
            get_prompt_for_step(bundle, "invalid_step")

    def test_all_prompts_have_content(self):
        """Test that all loaded prompts have non-empty content."""
        bundle = load_prompts_bundle()

        all_prompts = bundle.list_all_prompts()
        for name, prompt in all_prompts.items():
            assert prompt.content, f"Prompt {name} has empty content"
            assert (
                len(prompt.content) > 50
            ), f"Prompt {name} content seems too short"

    def test_all_prompts_have_descriptions(self):
        """Test that all loaded prompts have descriptions."""
        bundle = load_prompts_bundle()

        all_prompts = bundle.list_all_prompts()
        for name, prompt in all_prompts.items():
            assert prompt.description, f"Prompt {name} has no description"
            assert (
                len(prompt.description) > 10
            ), f"Prompt {name} description seems too short"

    def test_all_prompts_have_tags(self):
        """Test that all loaded prompts have at least one tag."""
        bundle = load_prompts_bundle()

        all_prompts = bundle.list_all_prompts()
        for name, prompt in all_prompts.items():
            assert prompt.tags, f"Prompt {name} has no tags"
            assert (
                len(prompt.tags) >= 1
            ), f"Prompt {name} should have at least one tag"
