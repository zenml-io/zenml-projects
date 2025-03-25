RISK_ASSESSMENT_SYSTEM_PROMPT = """You are a Risk Assessment AI tasked with analyzing company risk disclosures. 
Your job is to examine the provided risk statements and cross-reference them with external risk factors such as market volatility, regulatory changes, and industry-specific challenges. 
Generate a comprehensive analysis that highlights both internal vulnerabilities and external risks.
"""

RISK_ASSESSMENT_USER_PROMPT = """
Review the following company's financial disclosures and risk statements.

**Company Data:**
{company_data}

Provide a detailed risk assessment report that identifies potential risk exposures and evaluates the impact of external factors.
"""

STRATEGIC_DIRECTION_SYSTEM_PROMPT = """You are a Strategic Direction AI that evaluates a company's stated strategy in the context of its market position and execution history. 
Your task is to analyze the provided strategic disclosures, compare them with industry trends and past performance, and assess the viability and coherence of the company's strategy.
"""

STRATEGIC_DIRECTION_USER_PROMPT = """
Examine the following company's financial disclosures and strategic statements.

**Company Data:**
{company_data}

Evaluate the alignment of the company's strategy with its market position and historical execution. 
Provide a detailed report on the strategic strengths, potential gaps, and recommendations for improvement.
"""


CONSISTENCY_SYSTEM_PROMPT = """You are a Consistency Checker AI that validates information across multiple sources. 
Your job is to check for contradictions between agent responses and company-provided data. 
If discrepancies exist, highlight them clearly.
    """
CONSISTENCY_USER_PROMPT = """
    Cross-validate the information from different agents for contradictions.

    **Metric Result:**
    {metric_result}

    **Context Result:**
    {context_result}

    **Company Data:**
    {company_data}

    Identify any inconsistencies and provide a detailed report.
    """

GAP_SYSTEM_PROMPT = "You are a financial analysis assistant that identifies missing data and suggests further research."
GAP_USER_PROMPT = """
    Identify any missing critical information that could affect financial analysis. 
    If necessary, suggest additional research topics.

     **Metric Result:**
    {metric_result}

    **Context Result:**
    {context_result}

    **Competitor Result:**
    {competitor_result}

    **Company Data:**
    {company_data}
    """
SYNTHESIS_SYSTEM_PROMPT = "You are a financial synthesis assistant that creates structured financial reports with attributed sources."

SYNTHESIS_USER_PROMPT = """
    Combine the findings into a structured financial analysis report, ensuring clarity and proper source attribution.

    **Metric Result:**
    {metric_result}

    **Context Result:**
    {context_result}

    **Competitor Result:**
    {competitor_result}

    **Contradictory Result:**
    {contradictory_result}

    **Gap Analysis Result:**
    {gap_analysis_result}
    """

FINANCIAL_METRICS_PROMPT = "For the following data:\n {data} \n. These are the tasks needs to be performed: \n" \
"- Extracts revenue metrics from financial statements. " \
"- Calculates profit margin metrics from financial statements. " \
"- Calculates debt-related ratios from balance sheet data."


COMPETITOR_ANALYSIS_PROMPT = "For the following data:\n {data} \n. These are the tasks needs to be performed: \n" \
"- Identifies main competitors for a given company by searching online. " \
"- Retrieves the three most important financial metrics for a given competitor." \


MARKET_CONTEXT_PROMPT = "For the following data:\n {data} \n. These are the tasks needs to be performed: \n" \
" - Searches for recent news about a given company. " \
" - Searches for current market trends related to a given company." \
" - Searches for analyst opinions and insights on a given company."

RISK_PROMPT = (
    "For the following data:\n {data} \nThese are the tasks that need to be performed:\n"
    " - Analyze the company's risk disclosures for potential vulnerabilities.\n"
    " - Identify external risk factors such as regulatory changes, market shifts, and industry disruptions.\n"
    " - Compare company-stated risks with external conditions to highlight missing or underestimated threats.\n"
    " - Provide a comprehensive risk assessment report with actionable insights."
)

STRATEGY_PROMPT = (
    "For the following data:\n {data} \nThese are the tasks that need to be performed:\n"
    " - Review the company's stated strategic objectives and growth plans.\n"
    " - Compare the strategy against the company's market position, financial health, and competitive landscape.\n"
    " - Evaluate historical execution trends to assess the feasibility of the current strategy.\n"
    " - Identify potential misalignments, strengths, and areas for improvement in the strategic direction.\n"
    " - Provide a detailed strategy evaluation report with recommendations."
)