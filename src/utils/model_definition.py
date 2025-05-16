from pydantic import ConfigDict
from zenml import Model

model_definition = Model(
    name="credit_scoring_model",
    license="Apache 2.0",
    description="A credit scoring model",
    tags=["credit_scoring", "classifier"],
    audience="ZenML users",
    use_cases="EU AI Act compliance, risk assessment",
    model_config=ConfigDict(arbitrary_types_allowed=True),
)
