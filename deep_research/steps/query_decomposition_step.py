import logging
import time
from typing import Annotated

from materializers.query_context_materializer import QueryContextMaterializer
from utils.llm_utils import get_structured_llm_output
from utils.pydantic_models import Prompt, QueryContext
from utils.tracking_config import configure_tracking_provider
from zenml import log_metadata, step

logger = logging.getLogger(__name__)


@step(output_materializers=QueryContextMaterializer)
def initial_query_decomposition_step(
    main_query: str,
    query_decomposition_prompt: Prompt,
    llm_model: str = "openrouter/google/gemini-2.0-flash-lite-001",
    max_sub_questions: int = 8,
    tracking_provider: str = "weave",
    langfuse_project_name: str = "deep-research",
    weave_project_name: str = "deep-research",
) -> Annotated[QueryContext, "query_context"]:
    """Break down a complex research query into specific sub-questions.

    Args:
        main_query: The main research query to decompose
        query_decomposition_prompt: Prompt for query decomposition
        llm_model: The reasoning model to use with provider prefix
        max_sub_questions: Maximum number of sub-questions to generate
        tracking_provider: Experiment tracking provider (weave, langfuse, or none)
        langfuse_project_name: Langfuse project name for tracing
        weave_project_name: Weave project name for tracing

    Returns:
        QueryContext containing the main query and decomposed sub-questions
    """
    start_time = time.time()
    
    # Configure tracking provider
    project_name = langfuse_project_name if tracking_provider == "langfuse" else weave_project_name
    configure_tracking_provider(
        tracking_provider=tracking_provider,
        langfuse_project_name=langfuse_project_name,
        weave_project_name=weave_project_name,
    )
    
    logger.info(f"Decomposing research query: {main_query}")

    # Get the prompt content
    system_prompt = str(query_decomposition_prompt)

    try:
        # Call OpenAI API to decompose the query
        updated_system_prompt = (
            system_prompt
            + f"\nPlease generate at most {max_sub_questions} sub-questions."
        )
        logger.info(
            f"Calling {llm_model} to decompose query into max {max_sub_questions} sub-questions"
        )

        # Define fallback questions
        fallback_questions = [
            {
                "sub_question": f"What is {main_query}?",
                "reasoning": "Basic understanding of the topic",
            },
            {
                "sub_question": f"What are the key aspects of {main_query}?",
                "reasoning": "Exploring important dimensions",
            },
            {
                "sub_question": f"What are the implications of {main_query}?",
                "reasoning": "Understanding broader impact",
            },
        ]

        # Use utility function to get structured output
        decomposed_questions = get_structured_llm_output(
            prompt=main_query,
            system_prompt=updated_system_prompt,
            model=llm_model,
            fallback_response=fallback_questions,
            tracking_provider=tracking_provider,
            project=project_name,
        )

        # Extract just the sub-questions
        sub_questions = [
            item.get("sub_question")
            for item in decomposed_questions
            if "sub_question" in item
        ]

        # Limit to max_sub_questions
        sub_questions = sub_questions[:max_sub_questions]

        logger.info(f"Generated {len(sub_questions)} sub-questions")
        for i, question in enumerate(sub_questions, 1):
            logger.info(f"  {i}. {question}")

        # Create the QueryContext
        query_context = QueryContext(
            main_query=main_query, sub_questions=sub_questions
        )

        # Log step metadata
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "query_decomposition": {
                    "execution_time_seconds": execution_time,
                    "num_sub_questions": len(sub_questions),
                    "llm_model": llm_model,
                    "max_sub_questions_requested": max_sub_questions,
                    "fallback_used": False,
                    "main_query_length": len(main_query),
                    "sub_questions": sub_questions,
                }
            }
        )

        # Log model metadata for cross-pipeline tracking
        log_metadata(
            metadata={
                "research_scope": {
                    "num_sub_questions": len(sub_questions),
                }
            },
            infer_model=True,
        )

        # Log artifact metadata for the output query context
        log_metadata(
            metadata={
                "query_context_characteristics": {
                    "main_query": main_query,
                    "num_sub_questions": len(sub_questions),
                    "timestamp": query_context.decomposition_timestamp,
                }
            },
            infer_artifact=True,
        )

        # Add tags to the artifact
        # add_tags(tags=["query", "decomposed"], artifact_name="query_context", infer_artifact=True)

        return query_context

    except Exception as e:
        logger.error(f"Error decomposing query: {e}")
        # Return fallback questions
        fallback_questions = [
            f"What is {main_query}?",
            f"What are the key aspects of {main_query}?",
            f"What are the implications of {main_query}?",
        ]
        fallback_questions = fallback_questions[:max_sub_questions]
        logger.info(f"Using {len(fallback_questions)} fallback questions:")
        for i, question in enumerate(fallback_questions, 1):
            logger.info(f"  {i}. {question}")

        # Create QueryContext with fallback questions
        query_context = QueryContext(
            main_query=main_query, sub_questions=fallback_questions
        )

        # Log metadata for fallback scenario
        execution_time = time.time() - start_time
        log_metadata(
            metadata={
                "query_decomposition": {
                    "execution_time_seconds": execution_time,
                    "num_sub_questions": len(fallback_questions),
                    "llm_model": llm_model,
                    "max_sub_questions_requested": max_sub_questions,
                    "fallback_used": True,
                    "error_message": str(e),
                    "main_query_length": len(main_query),
                    "sub_questions": fallback_questions,
                }
            }
        )

        # Log model metadata for cross-pipeline tracking
        log_metadata(
            metadata={
                "research_scope": {
                    "num_sub_questions": len(fallback_questions),
                }
            },
            infer_model=True,
        )

        # Add tags to the artifact
        # add_tags(
        #     tags=["query", "decomposed", "fallback"], artifact_name="query_context", infer_artifact=True
        # )

        return query_context
