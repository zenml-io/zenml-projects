from typing import Dict, Any
from smolagents import tool
from langchain_community.utilities import SearchApiAPIWrapper
from dotenv import load_dotenv

load_dotenv()

search = SearchApiAPIWrapper(searchapi_api_key="sugLWZiekNUD55rLbni98E2R")

@tool
def search_recent_news(company_name: str) -> Dict[str, Any]:
    """
    Searches for recent news about a given company.
    
    Args:
        company_name: The name of the company to search for.
        
    Returns:
        Dictionary containing recent news articles.
    """
    search_query = f"{company_name} latest news"
    search_results = search.run(search_query)
    
    return {
        "company": company_name,
        "recent_news": search_results
    }

@tool
def search_market_trends(company_name: str) -> Dict[str, Any]:
    """
    Searches for current market trends related to a given company.
    
    Args:
        company_name: The name of the company to search for.
        
    Returns:
        Dictionary containing market trends.
    """
    search_query = f"{company_name} market trends"
    search_results = search.run(search_query)
    
    return {
        "company": company_name,
        "market_trends": search_results
    }

@tool
def search_analyst_opinions(company_name: str) -> Dict[str, Any]:
    """
    Searches for analyst opinions and insights on a given company.
    
    Args:
        company_name: The name of the company to search for.
        
    Returns:
        Dictionary containing analyst opinions and insights.
    """
    search_query = f"{company_name} analyst opinions"
    search_results = search.run(search_query)
    
    return {
        "company": company_name,
        "analyst_opinions": search_results
    }
