import os
from typing import Dict, List, Tuple

from agent.prompt import PREFIX, SUFFIX
from langchain.agents import AgentExecutor, ConversationalChatAgent
from langchain.schema.vectorstore import VectorStore
from langchain.tools.base import BaseTool
from langchain_community.tools.vectorstore.tool import VectorStoreQATool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing_extensions import Annotated
from zenml import ArtifactConfig, log_artifact_metadata, step
from zenml.client import Client
from zenml.enums import ArtifactType

PIPELINE_NAME = "zenml_agent_creation_pipeline"
# Choose what character to use for your agent's answers
CHARACTER = "technical assistant"

# First try to get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# If not found in env, fall back to ZenML secret
if not api_key:
    secret = Client().get_secret("llm_complete")
    api_key = secret.secret_values["openai_api_key"]


class AgentParameters(BaseModel):
    """Parameters for the agent."""

    llm: Dict = {
        "temperature": 0,
        "max_tokens": 1000,
        "model_name": "gpt-3.5-turbo",
        "api_key": api_key,
    }

    # allow extra fields
    class Config:
        extra = "ignore"


@step()
def agent_creator(
    vector_store: VectorStore, config: AgentParameters = AgentParameters()
) -> Annotated[
    Tuple[ConversationalChatAgent, List[BaseTool]],
    ArtifactConfig(name="agent", artifact_type=ArtifactType.DATA),
]:
    """Create an agent from a vector store.

    Args:
        vector_store: Vector store to create agent from.

    Returns:
        An AgentExecutor.
    """
    tools = [
        VectorStoreQATool(
            name=f"zenml-qa-tool",
            vectorstore=vector_store,
            description="Use this tool to answer questions about ZenML. "
            "How to debug errors in ZenML, how to answer conceptual "
            "questions about ZenML like available features, existing abstractions, "
            "and other parts from the documentation.",
            llm=ChatOpenAI(**config.llm),
        ),
    ]

    system_prompt = PREFIX.format(character=CHARACTER)

    my_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=ChatOpenAI(**config.llm),
        tools=tools,
        system_message=system_prompt,
        human_message=SUFFIX,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=my_agent,
        tools=tools,
        verbose=True,
    )

    log_artifact_metadata(
        artifact_name="agent",
        metadata={
            "Tools and their descriptions": {
                tool.name: tool.description for tool in tools
            },
            "Personality": {
                "character": CHARACTER,
                "temperature": config.llm["temperature"],
                "model_name": config.llm["model_name"],
            },
        },
    )

    return my_agent, tools
