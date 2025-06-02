"""Unit tests for prompt models and utilities."""

from utils.prompt_models import PromptTemplate
from utils.pydantic_models import Prompt


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


class TestPrompt:
    """Test cases for the new Prompt model."""

    def test_prompt_creation(self):
        """Test creating a prompt with all fields."""
        prompt = Prompt(
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

    def test_prompt_minimal(self):
        """Test creating a prompt with minimal fields."""
        prompt = Prompt(name="minimal_prompt", content="Minimal content")

        assert prompt.name == "minimal_prompt"
        assert prompt.content == "Minimal content"
        assert prompt.description == ""
        assert prompt.version == "1.0.0"
        assert prompt.tags == []

    def test_prompt_str_conversion(self):
        """Test converting prompt to string returns content."""
        prompt = Prompt(
            name="test_prompt",
            content="This is the prompt content",
            description="Test prompt",
        )

        assert str(prompt) == "This is the prompt content"

    def test_prompt_repr(self):
        """Test prompt representation."""
        prompt = Prompt(name="test_prompt", content="Content", version="2.0.0")

        assert repr(prompt) == "Prompt(name='test_prompt', version='2.0.0')"

    def test_prompt_create_factory(self):
        """Test creating prompt using factory method."""
        prompt = Prompt.create(
            content="Factory created prompt",
            name="factory_prompt",
            description="Created via factory",
            version="1.1.0",
            tags=["factory", "test"],
        )

        assert prompt.name == "factory_prompt"
        assert prompt.content == "Factory created prompt"
        assert prompt.description == "Created via factory"
        assert prompt.version == "1.1.0"
        assert prompt.tags == ["factory", "test"]

    def test_prompt_create_factory_minimal(self):
        """Test creating prompt using factory method with minimal args."""
        prompt = Prompt.create(
            content="Minimal factory prompt", name="minimal_factory"
        )

        assert prompt.name == "minimal_factory"
        assert prompt.content == "Minimal factory prompt"
        assert prompt.description == ""
        assert prompt.version == "1.0.0"
        assert prompt.tags == []
