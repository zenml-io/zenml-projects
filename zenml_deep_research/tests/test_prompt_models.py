"""Unit tests for prompt models and utilities."""

import pytest
from utils.prompt_models import PromptsBundle, PromptTemplate


class TestPromptTemplate:
    """Test cases for PromptTemplate model."""

    def test_prompt_template_creation(self):
        """Test creating a prompt template with all fields."""
        prompt = PromptTemplate(
            name="test_prompt",
            content="This is a test prompt",
            description="A test prompt for unit testing",
            version="1.0.0",
            tags=["test", "unit"],
        )

        assert prompt.name == "test_prompt"
        assert prompt.content == "This is a test prompt"
        assert prompt.description == "A test prompt for unit testing"
        assert prompt.version == "1.0.0"
        assert prompt.tags == ["test", "unit"]

    def test_prompt_template_minimal(self):
        """Test creating a prompt template with minimal fields."""
        prompt = PromptTemplate(
            name="minimal_prompt", content="Minimal content"
        )

        assert prompt.name == "minimal_prompt"
        assert prompt.content == "Minimal content"
        assert prompt.description == ""
        assert prompt.version == "1.0.0"
        assert prompt.tags == []


class TestPromptsBundle:
    """Test cases for PromptsBundle model."""

    @pytest.fixture
    def sample_prompts(self):
        """Create sample prompts for testing."""
        return {
            "search_query_prompt": PromptTemplate(
                name="search_query_prompt",
                content="Search query content",
                description="Generates search queries",
                tags=["search"],
            ),
            "query_decomposition_prompt": PromptTemplate(
                name="query_decomposition_prompt",
                content="Query decomposition content",
                description="Decomposes queries",
                tags=["analysis"],
            ),
            "synthesis_prompt": PromptTemplate(
                name="synthesis_prompt",
                content="Synthesis content",
                description="Synthesizes information",
                tags=["synthesis"],
            ),
            "viewpoint_analysis_prompt": PromptTemplate(
                name="viewpoint_analysis_prompt",
                content="Viewpoint analysis content",
                description="Analyzes viewpoints",
                tags=["analysis"],
            ),
            "reflection_prompt": PromptTemplate(
                name="reflection_prompt",
                content="Reflection content",
                description="Reflects on research",
                tags=["reflection"],
            ),
            "additional_synthesis_prompt": PromptTemplate(
                name="additional_synthesis_prompt",
                content="Additional synthesis content",
                description="Additional synthesis",
                tags=["synthesis"],
            ),
            "conclusion_generation_prompt": PromptTemplate(
                name="conclusion_generation_prompt",
                content="Conclusion generation content",
                description="Generates conclusions",
                tags=["report"],
            ),
        }

    def test_prompts_bundle_creation(self, sample_prompts):
        """Test creating a prompts bundle."""
        bundle = PromptsBundle(**sample_prompts)

        assert bundle.search_query_prompt.name == "search_query_prompt"
        assert (
            bundle.query_decomposition_prompt.name
            == "query_decomposition_prompt"
        )
        assert bundle.pipeline_version == "1.0.0"
        assert isinstance(bundle.created_at, str)
        assert bundle.custom_prompts == {}

    def test_prompts_bundle_with_custom_prompts(self, sample_prompts):
        """Test creating a prompts bundle with custom prompts."""
        custom_prompt = PromptTemplate(
            name="custom_prompt",
            content="Custom prompt content",
            description="A custom prompt",
        )

        bundle = PromptsBundle(
            **sample_prompts, custom_prompts={"custom_prompt": custom_prompt}
        )

        assert "custom_prompt" in bundle.custom_prompts
        assert bundle.custom_prompts["custom_prompt"].name == "custom_prompt"

    def test_get_prompt_by_name(self, sample_prompts):
        """Test retrieving prompts by name."""
        bundle = PromptsBundle(**sample_prompts)

        # Test getting a core prompt
        prompt = bundle.get_prompt_by_name("search_query_prompt")
        assert prompt is not None
        assert prompt.name == "search_query_prompt"

        # Test getting a non-existent prompt
        prompt = bundle.get_prompt_by_name("non_existent")
        assert prompt is None

    def test_get_prompt_by_name_custom(self, sample_prompts):
        """Test retrieving custom prompts by name."""
        custom_prompt = PromptTemplate(
            name="custom_prompt", content="Custom content"
        )

        bundle = PromptsBundle(
            **sample_prompts, custom_prompts={"custom_prompt": custom_prompt}
        )

        prompt = bundle.get_prompt_by_name("custom_prompt")
        assert prompt is not None
        assert prompt.name == "custom_prompt"

    def test_list_all_prompts(self, sample_prompts):
        """Test listing all prompts."""
        bundle = PromptsBundle(**sample_prompts)

        all_prompts = bundle.list_all_prompts()
        assert len(all_prompts) == 7  # 7 core prompts
        assert "search_query_prompt" in all_prompts
        assert "conclusion_generation_prompt" in all_prompts

    def test_list_all_prompts_with_custom(self, sample_prompts):
        """Test listing all prompts including custom ones."""
        custom_prompt = PromptTemplate(
            name="custom_prompt", content="Custom content"
        )

        bundle = PromptsBundle(
            **sample_prompts, custom_prompts={"custom_prompt": custom_prompt}
        )

        all_prompts = bundle.list_all_prompts()
        assert len(all_prompts) == 8  # 7 core + 1 custom
        assert "custom_prompt" in all_prompts

    def test_get_prompt_content(self, sample_prompts):
        """Test getting prompt content by type."""
        bundle = PromptsBundle(**sample_prompts)

        content = bundle.get_prompt_content("search_query_prompt")
        assert content == "Search query content"

        content = bundle.get_prompt_content("synthesis_prompt")
        assert content == "Synthesis content"

    def test_get_prompt_content_invalid(self, sample_prompts):
        """Test getting prompt content with invalid type."""
        bundle = PromptsBundle(**sample_prompts)

        with pytest.raises(AttributeError):
            bundle.get_prompt_content("invalid_prompt_type")
