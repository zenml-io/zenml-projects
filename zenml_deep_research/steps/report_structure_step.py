import json
import logging
import os
import openai
from typing import Annotated
from zenml import step
from materializers.state_visualizer import (
    StateMaterializer,
)

from utils.data_models import State, Paragraph
from utils.helper_functions import (
    remove_reasoning_from_output,
    clean_json_tags,
    safe_json_loads,
)

logger = logging.getLogger(__name__)

# System prompt for the report structure generation
REPORT_STRUCTURE_PROMPT = """
You are a Deep Research assistant. Given a query, plan a structure for a report and the paragraphs to be included.
Make sure that the ordering of paragraphs makes sense.
Once the outline is created, you will be given tools to search the web and reflect for each of the section separately.
Format the output in json with the following json schema definition:

<OUTPUT JSON SCHEMA>
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "content": {"type": "string"}
    }
  }
}
</OUTPUT JSON SCHEMA>

Title and content properties will be used for deeper research.
Make sure that the output is a json object with an output json schema defined above.
Only return the json object, no explanation or additional text.
"""


@step(output_materializers=StateMaterializer)
def report_structure_step(
    query: str = "What is ZenML?",
    sambanova_base_url: str = "https://api.sambanova.ai/v1",
    llm_model: str = "DeepSeek-R1-Distill-Llama-70B",
    system_prompt: str = REPORT_STRUCTURE_PROMPT,
) -> Annotated[State, "report_structure_step"]:
    """Generate the initial structure for a research report based on the query.

    Args:
        query: The research query/topic
        sambanova_base_url: SambaNova API base URL
        llm_model: The reasoning model to use
        system_prompt: System prompt for the LLM

    Returns:
        Initial state with report structure
    """
    logger.info(f"Generating report structure for query: {query}")

    # Get API key directly from environment variables
    sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY", "")
    if not sambanova_api_key:
        logger.error("SAMBANOVA_API_KEY environment variable not set")
        raise ValueError("SAMBANOVA_API_KEY environment variable not set")

    # Initialize OpenAI client
    openai_client = openai.OpenAI(
        api_key=sambanova_api_key, base_url=sambanova_base_url
    )

    try:
        # Call OpenAI API to generate report structure
        logger.info(f"Calling {llm_model} to generate report structure")
        response = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        )

        # Process the response
        content = response.choices[0].message.content
        content = remove_reasoning_from_output(content)
        content = clean_json_tags(content)

        # Parse the JSON
        report_structure = safe_json_loads(content)

        if not report_structure:
            logger.warning(
                "Failed to parse report structure, using fallback structure"
            )
            report_structure = [
                {
                    "title": "Introduction",
                    "content": "Introduction to the topic",
                },
                {
                    "title": "Main Findings",
                    "content": "Key findings about the topic",
                },
                {
                    "title": "Conclusion",
                    "content": "Concluding thoughts and summary",
                },
            ]

        # Create a descriptive report title from the query
        report_title = f"Research Report: {query}"

        # Create state with paragraphs
        state = State(report_title=report_title, query=query, paragraphs=[])

        for paragraph in report_structure:
            state.paragraphs.append(
                Paragraph(
                    title=paragraph.get("title", "Untitled Section"),
                    content=paragraph.get("content", ""),
                )
            )

        logger.info(
            f"Generated report structure with {len(state.paragraphs)} paragraphs"
        )

        return state

    except Exception as e:
        logger.error(f"Error generating report structure: {e}")
        # Return a minimal state with basic structure as fallback
        return State(
            report_title=f"Research Report: {query}",
            query=query,
            paragraphs=[
                Paragraph(
                    title="Introduction", content="Introduction to the topic"
                ),
                Paragraph(
                    title="Main Findings",
                    content="Key findings about the topic",
                ),
                Paragraph(
                    title="Conclusion",
                    content="Concluding thoughts and summary",
                ),
            ],
        )
