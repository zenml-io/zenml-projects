from typing import Dict, Any
from smolagents import tool

@tool
def extract_revenue_metrics(financial_statements: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts revenue metrics from financial statements.
    
    Args:
        financial_statements: Dictionary containing financial statement data
        
    Returns:
        Dictionary with revenue metrics including growth rates
    """
    revenue_data = financial_statements.get("income_statement", {}).get("revenue", {})
    current_revenue = revenue_data.get("current", 0)
    previous_revenue = revenue_data.get("previous", 0)
    
    growth_rate = ((current_revenue - previous_revenue) / previous_revenue) * 100 if previous_revenue > 0 else 0
    
    return {
        "current_revenue": current_revenue,
        "previous_revenue": previous_revenue,
        "revenue_growth_rate": growth_rate,
        "revenue_cagr_3yr": revenue_data.get("cagr_3yr", 0)
    }

@tool
def analyze_profit_margins(financial_statements: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates profit margin metrics from financial statements.
    
    Args:
        financial_statements: Dictionary containing financial statement data
        
    Returns:
        Dictionary with various profit margin metrics
    """
    income_statement = financial_statements.get("income_statement", {})
    revenue = income_statement.get("revenue", {}).get("current", 0)
    
    gross_profit = income_statement.get("gross_profit", 0)
    operating_income = income_statement.get("operating_income", 0)
    net_income = income_statement.get("net_income", 0)
    
    # Calculate margins
    gross_margin = (gross_profit / revenue) * 100 if revenue > 0 else 0
    operating_margin = (operating_income / revenue) * 100 if revenue > 0 else 0
    net_margin = (net_income / revenue) * 100 if revenue > 0 else 0
    
    return {
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "industry_avg_net_margin": income_statement.get("industry_avg_net_margin", 0)
    }

@tool
def calculate_debt_ratios(balance_sheet: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates debt-related ratios from balance sheet data.
    
    Args:
        balance_sheet: Dictionary containing balance sheet data
        
    Returns:
        Dictionary with debt ratios
    """
    total_assets = balance_sheet.get("total_assets", 0)
    total_debt = balance_sheet.get("total_debt", 0)
    total_equity = balance_sheet.get("total_equity", 0)
    current_assets = balance_sheet.get("current_assets", 0)
    current_liabilities = balance_sheet.get("current_liabilities", 1)  # Default to 1 to avoid division by zero
    
    debt_to_equity = total_debt / total_equity if total_equity > 0 else 0
    debt_to_assets = total_debt / total_assets if total_assets > 0 else 0
    current_ratio = current_assets / current_liabilities
    
    return {
        "debt_to_equity": debt_to_equity,
        "debt_to_assets": debt_to_assets,
        "current_ratio": current_ratio,
        "industry_avg_debt_to_equity": balance_sheet.get("industry_avg_debt_to_equity", 0)
    }