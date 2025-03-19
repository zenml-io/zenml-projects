# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Pydantic models for configuration validation.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from schemas import TrainingConfig

#
# Classification Pipeline Configs
#


class BatchProcessingConfig(BaseModel):
    """Configuration for batch processing parameters."""

    batch_start: int = Field(0, description="Index to start processing from")
    batch_size: int = Field(
        ..., description="Number of articles to process in this batch", gt=0
    )

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be greater than 0")
        return v


class ParallelProcessingConfig(BaseModel):
    """Configuration for parallel processing."""

    enabled: bool = Field(
        True, description="Whether to use parallel processing"
    )
    workers: int = Field(4, description="Number of parallel workers", ge=1)

    @field_validator("workers")
    @classmethod
    def validate_workers(cls, v):
        if v < 1:
            raise ValueError("Number of workers must be at least 1")
        return v


class InferenceParamsConfig(BaseModel):
    """Configuration for model inference parameters."""

    max_new_tokens: int = Field(
        ..., description="Maximum number of new tokens to generate"
    )
    max_sequence_length: int = Field(
        ..., description="Maximum sequence length for the model"
    )
    temperature: float = Field(
        ..., description="Temperature for sampling", ge=0.0, le=1.0
    )
    top_p: float = Field(
        ..., description="Top-p sampling parameter", ge=0.0, le=1.0
    )
    top_k: int = Field(..., description="Top-k sampling parameter", gt=0)

    @field_validator("temperature", "top_p")
    @classmethod
    def validate_probability_params(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                "Probability parameters must be between 0.0 and 1.0"
            )
        return v

    @model_validator(mode="after")
    def validate_token_lengths(self):
        if self.max_new_tokens >= self.max_sequence_length:
            raise ValueError(
                "max_new_tokens must be less than max_sequence_length"
            )
        return self


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint behavior."""

    enabled: bool = Field(
        default=True, description="Whether to enable checkpointing"
    )

    frequency: int = Field(
        default=10,
        description="Number of articles to process before saving a checkpoint",
        ge=1,
    )

    run_id: Optional[str] = Field(
        None,
        description="Unique identifier for the run if resuming from a specific run",
    )


class ClassificationPipelineConfig(BaseModel):
    """Configuration for the classification pipeline."""

    classification_type: Literal["evaluation", "augmentation"] = Field(
        default="evaluation", description="Type of classification to perform"
    )
    batch_processing: Optional[BatchProcessingConfig] = Field(
        None,
        description="Batch processing parameters, null to process entire dataset",
    )
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig,
        description="Checkpoint configuration",
    )
    parallel_processing: ParallelProcessingConfig = Field(
        default_factory=ParallelProcessingConfig,
        description="Parallel processing configuration",
    )
    inference_params: InferenceParamsConfig = Field(
        default_factory=InferenceParamsConfig,
        description="Model inference parameters",
    )


#
# Training Pipeline Configs
#


class DatasetSourceConfig(BaseModel):
    """Configuration for the test dataset source."""

    source_type: Literal["artifact", "disk"] = Field(
        ..., description="Source type for test data"
    )
    path: Optional[str] = Field(
        None,
        description="Path to test set on disk (used if source_type is 'disk')",
    )
    artifact_name: Optional[str] = Field(
        None,
        description="Name of artifact (used if source_type is 'artifact')",
    )
    version: Optional[int] = Field(
        None,
        description="Version of artifact (used if source_type is 'artifact')",
    )


#
# Model Comparison Pipeline Configs
#


class BatchSizesConfig(BaseModel):
    """Configuration for batch sizes in model comparison."""

    modernbert: int = Field(
        ..., description="Batch size for ModernBERT inference", gt=0
    )
    claude: int = Field(
        ..., description="Batch size for Claude API calls", gt=0
    )


class ClaudeHaikuCostConfig(BaseModel):
    """Configuration for Claude Haiku cost calculations."""

    input_cost_per_1k: float = Field(
        ..., description="Cost per 1K input tokens", ge=0.0
    )
    output_cost_per_1k: float = Field(
        ..., description="Cost per 1K output tokens", ge=0.0
    )


class CostsConfig(BaseModel):
    """Configuration for cost calculations."""

    claude_haiku: ClaudeHaikuCostConfig = Field(
        ..., description="Claude Haiku cost parameters"
    )


class ModelComparisonPipelineConfig(BaseModel):
    """Configuration for the model comparison pipeline."""

    dataset: DatasetSourceConfig = Field(
        ..., description="Test dataset configuration"
    )
    batch_sizes: BatchSizesConfig = Field(
        ..., description="Batch sizes for different models"
    )
    costs: CostsConfig = Field(..., description="Cost calculation parameters")


#
# Steps Configs
#


class DataSplitConfig(BaseModel):
    """Configuration for dataset splitting."""

    test_size: float = Field(
        ...,
        description="Proportion of data to use for testing",
        ge=0.0,
        le=1.0,
    )
    validation_size: float = Field(
        ...,
        description="Proportion of non-test data to use for validation",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="after")
    def validate_split_sizes(self):
        """Ensure split proportions are valid."""
        if self.test_size + (1 - self.test_size) * self.validation_size >= 0.9:
            raise ValueError(
                "Training set would be too small. Reduce test_size or validation_size."
            )
        return self


class FinetuneConfig(BaseModel):
    """Configuration for the finetuning step."""

    parameters: TrainingConfig = Field(
        ..., description="Model training parameters"
    )


class StepsConfig(BaseModel):
    """Configuration for the training pipeline."""

    classify: ClassificationPipelineConfig = Field(
        ..., description="Classification pipeline configuration"
    )
    data_split: DataSplitConfig = Field(
        ..., description="Dataset splitting configuration"
    )
    finetune_modernbert: FinetuneConfig = Field(
        ..., description="Model training parameters"
    )
    compare: ModelComparisonPipelineConfig = Field(
        ..., description="Model comparison configuration"
    )


#
# General Configs
#


class DatasetPathsConfig(BaseModel):
    """Configuration for file paths."""

    unclassified: str = Field(..., description="Path to unclassified dataset")
    composite: str = Field(..., description="Path to composite dataset")
    augmented: str = Field(..., description="Path to augmented dataset")


class OutputPathsConfig(BaseModel):
    """Configuration for output paths."""

    ft_model: str = Field(..., description="Path to save finetuned model")
    ft_tokenizer: str = Field(
        ..., description="Path to save finetuned tokenizer"
    )
    classifications: str = Field(
        "classification_results",
        description="Path to save classification results",
    )


class ModelRepoIdsConfig(BaseModel):
    """Configuration for model IDs."""

    deepseek: str = Field(..., description="DeepSeek model ID")
    modernbert_base_model: str = Field(
        ..., description="ModernBERT base model ID"
    )
    huggingface_repo: str = Field(
        ..., description="HuggingFace repository for deployment"
    )


class ProjectConfig(BaseModel):
    """Configuration for ZenML porject information."""

    name: str = Field(..., description="Project name")
    version: str = Field(..., description="Project version")
    description: Optional[str] = Field(None, description="Project description")
    tags: Optional[List[str]] = Field(None, description="Project tags")


class AppConfig(BaseModel):
    """Main application configuration."""

    datasets: DatasetPathsConfig
    outputs: OutputPathsConfig
    model_repo_ids: ModelRepoIdsConfig
    project: ProjectConfig
    steps: StepsConfig
    model_config = {
        "extra": "allow",
        "protected_namespaces": (),
    }


def validate_config(config: Dict) -> AppConfig:
    """
    Validate configuration dictionary against Pydantic models.

    Args:
        config: Raw configuration dictionary loaded from base_config.yaml

    Returns:
        Validated AppConfig instance

    Raises:
        ValidationError: If configuration is invalid
    """
    return AppConfig(**config)
