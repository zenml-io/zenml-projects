from dataclasses import dataclass
from typing import Annotated, Dict, List, Tuple, Type

from pydantic import BaseModel
from zenml import pipeline, step
from utils.llm_utils import process_input_with_retrieval

from pydantic_ai import Agent, RunContext

class RAGAnswer(BaseModel):
    """Answer to the question with references to the relevant documents"""
    answer: str
    # TODO support adding URLs. can be done easily by updating the retrieval fn

rag_agent = Agent(
    'openai:gpt-4o-mini',
    result_type=RAGAnswer,
    system_prompt=(
        'You are an agent that has access to a RAG tool that you can use to answer question'
        'you will get an answer from this tool that you can return to the user.'
    ),
)

@rag_agent.tool
async def get_rag_answer(ctx: RunContext, question: str) -> str:
    """Use the RAG tool to answer the question
    
    Args:
        question: The question to answer
    
    Returns:
        str: The answer
    """
    return process_input_with_retrieval(question)

from zenml.materializers.base_materializer import BaseMaterializer
import json
import dill

class PydanticAgentMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (Agent,)

    def load(self, data_type: Type[Agent]) -> Agent:
        # Load components separately
        with open(f"{self.uri}/system_prompt.txt", "r") as f:
            system_prompt = f.read()
        with open(f"{self.uri}/llm.txt", "r") as f:
            llm = f.read()
        with open(f"{self.uri}/tools.pkl", "rb") as f:
            tools = dill.load(f)
        
        # Reconstruct the agent
        agent = Agent(llm, result_type=RAGAnswer, system_prompt=system_prompt)
        agent._function_tools = tools
        return agent

    def save(self, data: Agent) -> None:
        # Save system prompt
        with open(f"{self.uri}/system_prompt.txt", "w") as f:
            f.write(str(data.system_prompt()))
        
        # Save LLM identifier
        with open(f"{self.uri}/llm.txt", "w") as f:
            f.write(data.model.name())
        
        # Save tools using dill
        with open(f"{self.uri}/tools.pkl", "wb") as f:
            dill.dump(data._function_tools, f)

# if __name__ == "__main__":
#     breakpoint()
#     rag_agent._function_tools
#     print(rag_agent.run_sync("What is zenml"))



@step(output_materializers=PydanticAgentMaterializer)
def agent_creator() -> Annotated[Agent, "rag-agent"]:
    return rag_agent

# @pipeline
# def simple():
#     agent_creator()

# if __name__ == "__main__":
#     simple()
