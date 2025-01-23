from dataclasses import dataclass
from typing import Annotated, Dict, List, Tuple

from pydantic import BaseModel
from zenml import step
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
async def get_rag_answer(question: str) -> str:
    """Use the RAG tool to answer the question
    
    Args:
        question: The question to answer
    
    Returns:
        str: The answer
    """
    return process_input_with_retrieval(question)

@step
def agent_creator() -> Annotated[Agent, "rag-agent"]:
    return rag_agent
