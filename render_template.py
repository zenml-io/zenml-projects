import yaml
from jinja2 import Template

# Load the template
with open("compliance/templates/annex_iv_template.j2", "r") as f:
    template_str = f.read()

# Create a Jinja template object
template = Template(template_str)

# Create dummy data (example)
dummy_data = {
    "pipeline": {
        "name": "Credit Scoring Model v1.2",
        "version": "1.2.0",
        "description": "AI system for evaluating loan applications",
        "previous_versions": ["1.0.0", "1.1.0"],
        "steps": [
            {
                "name": "Data preprocessing",
                "type": "transform",
                "description": "Clean and normalize input data",
            }
        ],
        "deployments": [{"type": "Docker container", "created_at": "2025-01-15"}],
    },
    "run": {
        "id": "run-123456",
        "code_reference": {"commit_sha": "abc123def456"},
        "metrics": {
            "accuracy": 0.92,
            "auc": 0.89,
            "fairness_metrics": {"disparity": 0.05, "demographic_parity": 0.97},
        },
        "metadata": {
            "model_type": "XGBoost ensemble",
            "optim_target": "F1 score with fairness constraints",
        },
    },
    "environment_variables": {"ORGANIZATION_NAME": "Example Corp."},
    "manual_inputs": {
        "design_assumptions": "System assumes complete financial history is available",
        "bias_mitigation": "Applied post-processing techniques to minimize gender and age bias",
    },
}


def to_yaml(value):
    """Return a YAML representation of the value."""
    return yaml.dump(value, default_flow_style=False)


template.environment.filters["to_yaml"] = to_yaml

# Render the template with the data
rendered = template.render(**dummy_data)

# Save the rendered output
with open("output/preview_generated_annex_iv_output.md", "w") as f:
    f.write(rendered)

print("Template rendered to output/preview_generated_annex_iv_output.md")
