import logging
from typing import Dict, cast

from agent.agent_executor_materializer import AgentExecutorMaterializer
from agent.prompt import PREFIX, SUFFIX
from langchain.agents import ConversationalChatAgent
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import VectorStore
from langchain.tools.vectorstore.tool import VectorStoreQATool
from langchain.agents import AgentExecutor
from zenml.steps import BaseParameters
from zenml import step


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
    vector_store: VectorStore, config: AgentParameters, version: str
) -> AgentExecutor:
    """Create an agent from a vector store.

    Args:
        vector_store: Vector store to create agent from.

    Returns:
        An AgentExecutor.
    """
    # check if an output for this step exists already.
    # if so, add a new tool to the agent. otherwise, create a new
    # agent.
    from zenml.client import Client

    pipeline_model = Client().get_pipeline(name_id_or_prefix=PIPELINE_NAME)

    existing_tools = []

    if pipeline_model.runs is not None:
        # get the last run
        last_run = pipeline_model.runs[0]
        try:
            # get the agent_creator step
            agent_creator_step = last_run.steps["agent_creator"]
            # get the output
            existing_agent = agent_creator_step.output.load()
            # cast existing agent to AgentExecutor
            existing_agent = cast(AgentExecutor, existing_agent)

            # get the tools from the existing agent
            existing_tools = existing_agent.tools

            logging.info("Found existing agent.")
        except ValueError:
            logging.info("No existing agent found.")

    tools = [
        *existing_tools,
        VectorStoreQATool(
            name=f"zenml-{version}",
            vectorstore=vector_store,
            description="Use this tool to answer questions about ZenML version "
            f"{version}. How to debug errors in ZenML, how to answer conceptual "
            "questions about ZenML like available features, existing abstractions, "
            "and other parts from the documentation.",
            llm=ChatOpenAI(**config.llm),
        ),
    ]

    my_agent = ConversationalChatAgent.from_llm_and_tools(
        llm=ChatOpenAI(**config.llm),
        tools=tools,
        system_message=PREFIX,
        human_message=SUFFIX,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=my_agent,
        tools=tools,
        verbose=True,
    )

    logging.info("About to return agent executor.")
    return agent_executor