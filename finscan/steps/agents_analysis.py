from zenml import step
from typing_extensions import Annotated
from smolagents import LiteLLMModel, CodeAgent
from tools.financial_metrics_tool import extract_revenue_metrics,analyze_profit_margins,calculate_debt_ratios
from tools.market_context_tool import search_analyst_opinions, search_recent_news, search_market_trends
from tools.competitor_analysis_tool import identify_competitors, retrieve_comparable_metrics
from tools.risk_assesment_tool import risk_assessment
from tools.strategic_direction_tool import strategic_direction
from zenml import step

GENERATION_MODEL = "gpt-4o"

@step
def financial_metric_agent(query: str) -> Annotated[str,"metric_result"]:

    tools=[extract_revenue_metrics, analyze_profit_margins, calculate_debt_ratios]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    financial_metric_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = financial_metric_agent(query)

    return result

@step
def market_context_agent(query: str)  -> Annotated[str,"context_result"]:
    
    tools=[search_analyst_opinions, search_recent_news, search_market_trends]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    market_context_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = market_context_agent(query)

    return result

@step
def competitor_analysis_agent(query: str)  -> Annotated[str,"competitor_result"]:
    
    tools=[identify_competitors, retrieve_comparable_metrics]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    competitor_analysis_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = competitor_analysis_agent(query)

    return result

@step
def risk_assesment_agent(query: str)  -> Annotated[str,"risk_assesment_result"]:
    
    tools=[risk_assessment]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    risk_assesment_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = risk_assesment_agent(query)

    return result

@step
def strategic_direction_agent(query: str) -> Annotated[str,"strategic_direction_result"]:
    
    tools = [strategic_direction]
    financial_model = LiteLLMModel(model_id=GENERATION_MODEL)
    strategic_direction_agent = CodeAgent(
        tools=tools, 
        model=financial_model
    )
    result = strategic_direction_agent(query)

    return result

