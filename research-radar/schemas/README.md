# Schemas Directory

This directory contains Pydantic models that define data structures and validation rules used throughout various pipelines.

## Core Schemas

### [`classification_output.py`](classification_output.py)

Defines the schema for LLM-generated classification results:

- `is_accepted`: Boolean indicating article acceptance
- `confidence`: Float between 0.0-1.0 representing classification confidence
- `reason`: String explanation for the classification decision

### [`claude_response.py`](claude_response.py)

Schema for structured responses from the Claude API:

- `prediction`: Integer (0 or 1) representing article acceptance
- Captures latency, token counts, and raw response
- Includes error handling properties
- Provides utility methods for calculating total tokens and cost

> **Note:** While `ClassificationOutput` uses a boolean `is_accepted` field, `ClaudeResponse` uses an integer `prediction` field. The normalization from boolean to integer happens in the [`compare_models.py`](../steps/compare_models.py) module.

### [`input_article.py`](input_article.py)

Two-part schema for article data:

- `ArticleMeta`: URL, title, publication date, and author information
- `InputArticle`: Article text with metadata and validation rules
  - Ensures text is non-empty with field validation

### [`training_config.py`](training_config.py)

Configuration schema for Hugging Face `TrainingArguments`:

- Defines hyperparameters with sensible defaults and constraints
- Includes conversion method to Hugging Face `TrainingArguments` object
- Supports creation from dictionary configuration

### [`zenml_project.py`](zenml_project.py)

Utility for creating ZenML model configurations:

- Loads model metadata from settings
- Creates properly configured ZenML `Model` objects for tracking
- Sets name, version, description, license, and tags
