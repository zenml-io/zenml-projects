"""Step to initialize and track prompts as artifacts.

This step creates a PromptsBundle artifact at the beginning of the pipeline,
making all prompts trackable and versioned in ZenML.
"""

import logging
from typing import Annotated

from materializers.prompts_materializer import PromptsBundleMaterializer
from utils.prompt_loader import load_prompts_bundle
from utils.prompt_models import PromptsBundle
from zenml import ArtifactConfig, step

logger = logging.getLogger(__name__)


@step(output_materializers=PromptsBundleMaterializer)
def initialize_prompts_step(
    pipeline_version: str = "1.1.0",
) -> Annotated[
    PromptsBundle,
    ArtifactConfig(name="prompts_bundle", tags=["prompts", "configuration"]),
]:
    """Initialize the prompts bundle for the pipeline.

    This step loads all prompts from the prompts.py module and creates
    a PromptsBundle artifact that can be tracked and visualized in ZenML.

    Args:
        pipeline_version: Version of the pipeline using these prompts

    Returns:
        PromptsBundle containing all prompts used in the pipeline
    """
    logger.info(
        f"Initializing prompts bundle for pipeline version {pipeline_version}"
    )

    # Load all prompts into a bundle
    prompts_bundle = load_prompts_bundle(pipeline_version=pipeline_version)

    # Log some statistics
    all_prompts = prompts_bundle.list_all_prompts()
    logger.info(f"Loaded {len(all_prompts)} prompts into bundle")
    logger.info(f"Prompts: {', '.join(all_prompts.keys())}")

    return prompts_bundle
