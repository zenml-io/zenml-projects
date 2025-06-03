import json
import logging
import os
from typing import Annotated

from anthropic import Anthropic
from utils.prompts import MCP_PROMPT
from utils.pydantic_models import (
    AnalysisData,
    MCPResult,
    QueryContext,
    SynthesisData,
)
from zenml import step

logger = logging.getLogger(__name__)

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
exa_api_key = os.getenv("EXA_API_KEY")


def preprocess_data_for_prompt(
    query_context, synthesis_data: SynthesisData, analysis_data: AnalysisData
) -> dict:
    """Preprocess Pydantic objects into JSON strings for prompt injection.

    Args:
        query_context: Either a QueryContext object or a string containing the query
        synthesis_data: The synthesis data containing synthesized and enhanced info
        analysis_data: The analysis data containing viewpoint analysis and reflection metadata

    Returns:
        A dictionary with preprocessed JSON strings ready for prompt formatting
    """
    # Handle query_context - either string or QueryContext object
    if isinstance(query_context, str):
        user_query = query_context
        logger.warning("query_context is a string, not a QueryContext object")
    else:
        user_query = query_context.main_query

    # Convert synthesized_info dict to formatted JSON
    synthesized_info_json = json.dumps(
        {
            k: v.model_dump()
            for k, v in synthesis_data.synthesized_info.items()
        },
        indent=2,
        ensure_ascii=False,
    )

    # Convert enhanced_info dict to formatted JSON
    enhanced_info_json = json.dumps(
        {k: v.model_dump() for k, v in synthesis_data.enhanced_info.items()},
        indent=2,
        ensure_ascii=False,
    )

    # Convert viewpoint_analysis to formatted JSON (handle None case)
    if analysis_data.viewpoint_analysis:
        viewpoint_analysis_json = json.dumps(
            analysis_data.viewpoint_analysis.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    else:
        viewpoint_analysis_json = "No viewpoint analysis available"

    # Convert reflection_metadata to formatted JSON (handle None case)
    if analysis_data.reflection_metadata:
        reflection_metadata_json = json.dumps(
            analysis_data.reflection_metadata.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    else:
        reflection_metadata_json = "No reflection metadata available"

    return {
        "user_query": user_query,
        "synthesized_info": synthesized_info_json,
        "enhanced_info": enhanced_info_json,
        "viewpoint_analysis": viewpoint_analysis_json,
        "reflection_metadata": reflection_metadata_json,
    }


@step
def mcp_updates_step(
    query_context: QueryContext,
    synthesis_data: SynthesisData,
    analysis_data: AnalysisData,
    langfuse_project_name: str,
) -> Annotated[MCPResult, "mcp_results"]:
    """Additional MCP-driven search of Exa.

    This step is used to update the synthesis and analysis data with the results of
    the MCP tools.
    """
    try:
        # Preprocess all data into JSON strings for the prompt
        preprocessed_data = preprocess_data_for_prompt(
            query_context, synthesis_data, analysis_data
        )

        prompt = MCP_PROMPT.format(**preprocessed_data)

        logger.info(f"Making MCP request for query: {prompt}")

        response = anthropic.beta.messages.create(
            model="claude-sonnet-4-20250514",  # latest version works with MCP
            max_tokens=3000,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            mcp_servers=[
                {
                    "type": "url",
                    "url": f"https://mcp.exa.ai/mcp?exaApiKey={exa_api_key}",
                    "name": "exa-mcp-search",
                    "authorization_token": exa_api_key,
                    "tool_configuration": {
                        "enabled": True,
                        "allowed_tools": [
                            "research_paper_search",
                            "company_research",
                            "competitor_finder",
                            "linkedin_search",
                            "wikipedia_search_exa",
                            "github_search",
                        ],
                    },
                }
            ],
            betas=["mcp-client-2025-04-04"],  # needed for MCP functionality
        )

        # Safe extraction of response content
        raw_mcp_result = ""
        mcp_result = ""
        last_text_index = -1
        mcp_tool_result_index = -1

        if hasattr(response, "content") and response.content:
            # First pass: find indices of relevant content
            for i, content_item in enumerate(response.content):
                logger.debug(
                    f"Content item {i}: type={getattr(content_item, 'type', 'unknown')}"
                )

                if hasattr(content_item, "type"):
                    if content_item.type == "mcp_tool_result":
                        mcp_tool_result_index = i
                    elif content_item.type == "text":
                        last_text_index = i

            # Extract MCP tool result if found
            if mcp_tool_result_index >= 0:
                content_item = response.content[mcp_tool_result_index]
                if (
                    hasattr(content_item, "content")
                    and content_item.content
                    and len(content_item.content) > 0
                    and hasattr(content_item.content[0], "text")
                ):
                    raw_mcp_result = content_item.content[0].text
                    logger.info(
                        f"Found raw MCP result at index {mcp_tool_result_index}"
                    )

            # Extract the last text result if found
            if last_text_index >= 0:
                content_item = response.content[last_text_index]
                if hasattr(content_item, "text"):
                    mcp_result = content_item.text
                    logger.info(
                        f"Found MCP text result at index {last_text_index} (last text item)"
                    )

        if not raw_mcp_result and not mcp_result:
            logger.warning("No MCP results found in response")

        return MCPResult(
            raw_mcp_result=raw_mcp_result,
            mcp_result=mcp_result,
        )

    except Exception as e:
        logger.error(f"Error in MCP updates step: {str(e)}", exc_info=True)
        # Return empty results on error rather than crashing
        return MCPResult(
            raw_mcp_result="",
            mcp_result=f"Error occurred: {str(e)}",
        )
