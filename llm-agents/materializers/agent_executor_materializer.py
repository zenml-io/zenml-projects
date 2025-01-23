#  Copyright (c) ZenML GmbH 2024. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.
"""Implementation of ZenML's pickle materializer."""

import os
from typing import Any, ClassVar, Tuple, Type

import pickle
import json

from zenml.enums import ArtifactType
from zenml.environment import Environment
from zenml.io import fileio
from zenml.logger import get_logger
from zenml.materializers.base_materializer import BaseMaterializer
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools.base import BaseTool
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings


logger = get_logger(__name__)


class AgentExecutorMaterializer(BaseMaterializer):
    """Materializer for AgentExecutor."""

    ASSOCIATED_TYPES = (AgentExecutor, )

    def save(self, data: AgentExecutor) -> None:
        """Save the AgentExecutor by serializing its components.
        
        Args:
            data: The AgentExecutor to save
        """
        # Extract the components we need to save
        agent_config = {
            "llm_config": {
                "model_name": data.agent.llm.model_name,
                "temperature": data.agent.llm.temperature,
                # Add other LLM configs as needed
            },
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "return_direct": getattr(tool, "return_direct", False),
                }
                for tool in data.tools
            ],
            "prompt": data.agent.prompt.template if hasattr(data.agent, "prompt") else None,
        }

        # Save the configuration
        config_path = os.path.join(self.uri, "agent_config.json")
        with open(config_path, "w") as f:
            json.dump(agent_config, f)

        # Save the vector store if it exists in tools
        for tool in data.tools:
            if hasattr(tool, "vectorstore") and isinstance(tool.vectorstore, FAISS):
                vector_store_path = os.path.join(self.uri, "vector_store")
                tool.vectorstore.save_local(vector_store_path)
                break

    def load(self, data_type: Type[AgentExecutor]) -> AgentExecutor:
        """Load the AgentExecutor by reconstructing from saved components.
        
        Args:
            data_type: The type of the data to load

        Returns:
            The reconstructed AgentExecutor
        """
        # Load the configuration
        config_path = os.path.join(self.uri, "agent_config.json")
        with open(config_path, "r") as f:
            agent_config = json.load(f)

        # Reconstruct LLM
        llm = ChatOpenAI(**agent_config["llm_config"])

        # Reconstruct tools
        tools = []
        vector_store_path = os.path.join(self.uri, "vector_store")
        if os.path.exists(vector_store_path):
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.load_local(
                vector_store_path, 
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Reconstruct tools with vector store
            from langchain.tools import VectorStoreQA
            tools = [
                VectorStoreQA(
                    name=tool["name"],
                    description=tool["description"],
                    vectorstore=vector_store,
                    llm=llm,
                )
                for tool in agent_config["tools"]
            ]

        # Reconstruct prompt if it exists
        prompt = None
        if agent_config["prompt"]:
            prompt = PromptTemplate.from_template(agent_config["prompt"])

        # Reconstruct the agent
        from langchain.agents import initialize_agent
        agent_executor = initialize_agent(
            tools,
            llm,
            agent="chat-conversational-react-description",
            verbose=True,
            handle_parsing_errors=True,
        )

        return agent_executor
