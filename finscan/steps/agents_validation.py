from zenml import step
from typing_extensions import Annotated
from smolagents import LiteLLMModel, CodeAgent
from tools.consistency_tool import consistency_checker
from tools.gap_analysis_tool import gap_analysis
from tools.synthesis_tool import synthesis
from zenml import step

GENERATION_MODEL = "gpt-4o"

@step
def consistency_agent(query: str) -> Annotated[str,"consistency_result"]:

    tools=[consistency_checker]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    consistency_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = consistency_agent(query)

    return result

@step
def gap_agent(query: str) -> Annotated[str,"gap_result"]:

    tools=[gap_analysis]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    gap_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = gap_agent(query)

    return result

@step
def synthesis_agent(query: str) -> Annotated[str,"synthesis_result"]:
    tools=[synthesis]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    synthesis_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = synthesis_agent(query)

    return result