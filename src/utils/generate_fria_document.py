# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import Dict


def generate_fria_document(evaluation_results: Dict, risk_assessment: Dict):
    """Generate a Fundamental Rights Impact Assessment document."""
    template_path = "compliance/templates/fria_template.md"
    output_path = "reports/fria_assessment.md"

    with open(template_path, "r") as f:
        template_content = f.read()

    # Extract fairness metrics
    fairness_concerns = []
    for attr, metrics in evaluation_results["fairness_metrics"].items():
        if metrics["disparity"] < 0.8 or metrics["disparity"] > 1.25:
            fairness_concerns.append(
                f"- Potential bias detected for {attr}: disparity ratio = {metrics['disparity']:.2f}"
            )

    # Populate template
    fria_content = template_content.replace(
        "[FAIRNESS_METRICS]",
        "\n".join(fairness_concerns) or "No significant fairness concerns detected.",
    )

    # Add risk assessment
    risk_content = f"## Overall Risk Assessment\n\nRisk Level: {risk_assessment['risk_level']}\n\n"
    risk_content += "### Mitigation Measures\n\n"
    for measure in risk_assessment["mitigation_measures"]:
        risk_content += (
            f"- {measure['risk_area']}: {measure['measure']} ({measure['implementation_status']})\n"
        )

    fria_content = fria_content.replace("[RISK_ASSESSMENT]", risk_content)

    # Save document
    with open(output_path, "w") as f:
        f.write(fria_content)

    return fria_content
