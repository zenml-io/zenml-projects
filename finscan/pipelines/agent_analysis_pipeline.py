from zenml import pipeline, ArtifactConfig
from steps.agents_analysis import financial_metric_agent, market_context_agent, competitor_analysis_agent, risk_assesment_agent, strategic_direction_agent
from typing_extensions import Annotated
from typing import Dict
import os
import base64
from dotenv import load_dotenv
from opentelemetry.sdk.trace import TracerProvider
from langfuse import Langfuse
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from config.prompts import COMPETITOR_ANALYSIS_PROMPT, FINANCIAL_METRICS_PROMPT, MARKET_CONTEXT_PROMPT, RISK_PROMPT, STRATEGY_PROMPT
from zenml.artifacts.utils import save_artifact

load_dotenv()
 
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY") 
LANGFUSE_SECRET_KEY= os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_AUTH=base64.b64encode(f"{LANGFUSE_PUBLIC_KEY}:{LANGFUSE_SECRET_KEY}".encode()).decode()
 
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "https://cloud.langfuse.com/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

langfuse = Langfuse(
  secret_key=LANGFUSE_SECRET_KEY,
  public_key=LANGFUSE_SECRET_KEY,
  host="https://cloud.langfuse.com"
)

@pipeline()
def agent_analysis_pipeline(data: Dict):
    """
    ZenML pipeline that orchestrates document loading, processing, and storing.
    """
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
    metric_query = MARKET_CONTEXT_PROMPT.format(data = data)
    context_query = FINANCIAL_METRICS_PROMPT.format(data = data)
    competitor_query = COMPETITOR_ANALYSIS_PROMPT.format(data = data)
    risk_query = RISK_PROMPT.format(data = data)
    strategy_query = STRATEGY_PROMPT.format(data = data)
    metric_result = financial_metric_agent(metric_query)
    context_result = market_context_agent(context_query)
    competitor_result = competitor_analysis_agent(competitor_query)
    risk_result = risk_assesment_agent(risk_query)
    strategy_result = strategic_direction_agent(strategy_query)
    # result =  {"metric_result": metric_result, "context_result": context_result, "competitor_result": competitor_result, "risk_result":risk_result, "strategy_result": strategy_result}
    # #save_artifact(result, name="analysis_result")




