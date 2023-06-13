from typing import Dict
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import VectorStore, Weaviate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.tools.vectorstore.tool import VectorStoreQATool
from langchain.memory.vectorstore import VectorStoreRetrieverMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentExecutor, AgentType
from langchain.tools.base import BaseTool
from langchain.llms.base import LLM
from pydantic import BaseModel
from weaviate import Client
from zenml.steps import step, BaseParameters


class AgentParameters(BaseParameters):
    """Parameters for the agent."""

    llm: Dict = {
        "repo_id": "google/flan-t5-xl",
        "huggingfacehub_api_token": "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",
        "model_kwargs": {"temperature": 0, "max_length": 500},
    }

    weaviate_settings: Dict = {
        "url": "",
    }

    # allow extra fields
    class Config:
        extra = "ignore"


@step
def agent_creator(
    vector_store: VectorStore, config: AgentParameters
) -> AgentExecutor:
    """Create an agent from a vector store.

    Args:
        vector_store: Vector store to create agent from.

    Returns:
        An AgentExecutor.
    """
    tools = [
        VectorStoreQATool(
            name="ZenML",
            vectorstore=vector_store,
            description="How to debug errors in ZenML, how to answer conceptual "
            "questions about ZenML like available features, existing abstractions, "
            "and other parts from the documentation.",
            llm=HuggingFaceHub(**config.llm),
        ),
    ]

    retriever = VectorStoreRetriever(
        vectorstore=Weaviate(
            index_name="chat_history",
            client=Client(config.weaviate_settings["url"]),
        ),
    )

    memory = VectorStoreRetrieverMemory(
        retriever=retriever,
        memory_key="chat_history",
        return_docs=True,
    )

    agent_executor = initialize_agent(
        tools=tools,
        llm=config.llm,
        memory=memory,
        verbose=True,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    )

    return agent_executor
