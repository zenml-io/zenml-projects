from typing import Dict, Any
from smolagents import tool
from langchain_community.utilities import SearchApiAPIWrapper
from dotenv import load_dotenv
load_dotenv()

search = SearchApiAPIWrapper(searchapi_api_key="sugLWZiekNUD55rLbni98E2R")


@tool
def identify_competitors(company_name: str) -> Dict[str, Any]:
    """
    Identifies main competitors for a given company by searching online.
    
    Args:
        company_name: The name of the company to search competitors for.
    
    Returns:
        Dictionary containing a list of competitor names and raw search results.
    """
    search_query = f"{company_name} main competitors"
    search_results = search.run(search_query)
    
    return {
        "company": company_name,
        "competitors": search_results
    }

@tool
def retrieve_comparable_metrics(company_name: str) -> Dict[str, Any]:
    """
    Retrieves the three most important financial metrics for a given competitor.
    
    The metrics include:
      - Revenue Growth Rate
      - Net Margin
      - Debt-to-Equity Ratio
    
    Args:
        company_name: The name of the competitor company.
    
    Returns:
        Dictionary containing the competitor's key financial metrics and raw search results.
    """
    search_query = f"{company_name} revenue growth rate, net margin, debt to equity ratio"
    search_results = search.run(search_query)
    
    return {
        "company": company_name,
        "key_metrics": search_results
    }