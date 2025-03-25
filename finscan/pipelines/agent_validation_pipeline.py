from zenml import pipeline, ArtifactConfig
from steps.agents_validation import consistency_agent, gap_agent, synthesis_agent
from steps.report_dashboard import generate_metrics_html
from typing_extensions import Annotated
from typing import Dict, Any
import os
import json
import base64
from dotenv import load_dotenv
from opentelemetry.sdk.trace import TracerProvider
from langfuse import Langfuse
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

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
def agent_validation_pipeline(agent_responses: Dict, data: Dict) -> Annotated[Dict[str, Any], "validated_report"]:
    """
    ZenML pipeline that orchestrates document loading, processing, and storing.
    """
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
    SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    print(agent_responses, data)
    consistency_query = f"""I want you to validate the consistency of the following:
    Metric Results: {agent_responses['metric_result']}
    Context Result: {agent_responses['context_result']}
    Competitor Result: {agent_responses['competitor_result']}
    Company Data: {data}
    """
    gap_query = f"""I want you to check for any gaps in the following information:
    Metric Results: {agent_responses['metric_result']}
    Context Result: {agent_responses['context_result']}
    Competitor Result: {agent_responses['competitor_result']}
    Company Data:{data} 
    """
    contradictory_result = consistency_agent(consistency_query)
    gap_result = gap_agent(gap_query)
    synthesis_query = f"""I want you to consolidate the results from the following sources into a unified report:
    Metric Results: {agent_responses['metric_result']}
    Context Result: {agent_responses['context_result']}
    Competitor Result: {agent_responses['competitor_result']}
    Contradictory Analysis: {contradictory_result}
    Gap Analysis:{gap_result}
    """
    synthesis_result = synthesis_agent(synthesis_query)