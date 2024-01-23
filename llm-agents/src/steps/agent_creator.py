import logging
from typing import Annotated, Dict, cast

from agent.agent_executor_materializer import AgentExecutorMaterializer
from agent.prompt import PREFIX, SUFFIX
from langchain.agents import ConversationalChatAgent
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import VectorStore
from langchain.tools.vectorstore.tool import VectorStoreQATool
from langchain.agents import AgentExecutor
from zenml.steps import BaseParameters
from zenml import step, log_artifact_metadata


PIPELINE_NAME = "zenml_agent_creation_pipeline"


class AgentParameters(BaseParameters):
    """Parameters for the agent."""

    llm: Dict = {
        "temperature": 0,
        "max_tokens": 1000,
        "model_name": "gpt-3.5-turbo",
    }

    # allow extra fields
    class Config:
        extra = "ignore"


@step(output_materializers=AgentExecutorMaterializer, enable_cache=False)
def agent_creator(
    vector_store: VectorStore, config: AgentParameters
) -> Annotated[AgentExecutor, "agent"]:
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

    character = "technical assistant"
    # make a system prompt that uses the PREFIX and answers questions based
    # on the character's style
    system_prompt = PREFIX.format(character=character)

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
                "character": character,
                "temperature": config.llm["temperature"],
                "model_name": config.llm["model_name"],
            },
        }
    )

    return agent_executor
