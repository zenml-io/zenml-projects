"""Materializer for SynthesisData with confidence metrics and synthesis quality visualization."""

import os
from typing import Dict

from utils.css_utils import (
    create_stat_card,
    get_card_class,
    get_confidence_class,
    get_grid_class,
    get_shared_css_tag,
)
from utils.pydantic_models import SynthesisData
from zenml.enums import ArtifactType, VisualizationType
from zenml.io import fileio
from zenml.materializers import PydanticMaterializer


class SynthesisDataMaterializer(PydanticMaterializer):
    """Materializer for SynthesisData with quality metrics visualization."""

    ASSOCIATED_TYPES = (SynthesisData,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.DATA

    def save_visualizations(
        self, data: SynthesisData
    ) -> Dict[str, VisualizationType]:
        """Create and save visualizations for the SynthesisData.

        Args:
            data: The SynthesisData to visualize

        Returns:
            Dictionary mapping file paths to visualization types
        """
        visualization_path = os.path.join(self.uri, "synthesis_data.html")
        html_content = self._generate_visualization_html(data)

        with fileio.open(visualization_path, "w") as f:
            f.write(html_content)

        return {visualization_path: VisualizationType.HTML}

    def _generate_visualization_html(self, data: SynthesisData) -> str:
        """Generate HTML visualization for the synthesis data.

        Args:
            data: The SynthesisData to visualize

        Returns:
            HTML string
        """
        # Count confidence levels
        confidence_counts = {"high": 0, "medium": 0, "low": 0}
        for info in data.synthesized_info.values():
            confidence_counts[info.confidence_level] += 1

        # Create synthesis cards HTML
        synthesis_html = ""
        for sub_q, info in data.synthesized_info.items():
            sources_html = ""
            if info.key_sources:
                sources_html = "<div class='synthesis-sources'><strong>Key Sources:</strong><ul class='dr-list'>"
                for source in info.key_sources[:3]:  # Show first 3 sources
                    sources_html += f"<li>{source}</li>"
                if len(info.key_sources) > 3:
                    sources_html += (
                        f"<li>... and {len(info.key_sources) - 3} more</li>"
                    )
                sources_html += "</ul></div>"

            gaps_html = ""
            if info.information_gaps:
                gaps_html = f"""
                <div class='dr-notice dr-notice--warning'>
                    <strong>Information Gaps:</strong>
                    <p>{info.information_gaps}</p>
                </div>
                """

            improvements_html = ""
            if info.improvements:
                improvements_html = "<div class='synthesis-improvements'><strong>Suggested Improvements:</strong><ul class='dr-list'>"
                for imp in info.improvements:
                    improvements_html += f"<li>{imp}</li>"
                improvements_html += "</ul></div>"

            # Check if this has enhanced version
            enhanced_badge = ""
            enhanced_section = ""
            if sub_q in data.enhanced_info:
                enhanced_badge = (
                    '<span class="dr-badge dr-badge--primary">Enhanced</span>'
                )
                enhanced_info = data.enhanced_info[sub_q]
                enhanced_section = f"""
                <div class="enhanced-section">
                    <h4>Enhanced Answer</h4>
                    <p class="synthesis-answer">{enhanced_info.synthesized_answer}</p>
                    <div class="{get_confidence_class(enhanced_info.confidence_level)}">
                        Confidence: {enhanced_info.confidence_level.upper()}
                    </div>
                </div>
                """

            synthesis_html += f"""
            <div class="{get_card_class()}">
                <div class="dr-flex-between dr-mb-md">
                    <h3>{sub_q}</h3>
                    {enhanced_badge}
                </div>
                
                <div class="original-section">
                    <h4>Original Synthesis</h4>
                    <p class="synthesis-answer">{info.synthesized_answer}</p>
                    
                    <div class="{get_confidence_class(info.confidence_level)}">
                        Confidence: {info.confidence_level.upper()}
                    </div>
                    
                    {sources_html}
                    {gaps_html}
                    {improvements_html}
                </div>
                
                {enhanced_section}
            </div>
            """

        if not synthesis_html:
            synthesis_html = (
                '<div class="dr-empty">No synthesis data available yet</div>'
            )

        # Calculate statistics
        total_syntheses = len(data.synthesized_info)
        total_enhanced = len(data.enhanced_info)
        avg_sources = sum(
            len(info.key_sources) for info in data.synthesized_info.values()
        ) / max(total_syntheses, 1)

        # Create stats HTML
        stats_html = f"""
        <div class="{get_grid_class("stats")}">
            {create_stat_card(total_syntheses, "Total Syntheses")}
            {create_stat_card(total_enhanced, "Enhanced Syntheses")}
            {create_stat_card(f"{avg_sources:.1f}", "Avg Sources per Synthesis")}
            {create_stat_card(confidence_counts["high"], "High Confidence")}
        </div>
        """

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synthesis Data Visualization</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            {get_shared_css_tag()}
            <style>
                /* Component-specific styles */
                .synthesis-sources {{
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid var(--color-border);
                }}
                
                .synthesis-improvements {{
                    margin-top: 20px;
                    padding: 15px;
                    background: #f6fff9;
                    border-radius: var(--radius-md);
                    border-left: 4px solid var(--color-success);
                }}
                
                .original-section,
                .enhanced-section {{
                    margin-bottom: 20px;
                    padding: 20px;
                    background: var(--color-bg-secondary);
                    border-radius: var(--radius-md);
                }}
                
                .enhanced-section {{
                    background: #f0f5ff;
                    border: 2px solid var(--color-secondary);
                }}
                
                .synthesis-answer {{
                    line-height: 1.8;
                    color: var(--color-text-primary);
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="dr-container dr-container--wide">
                <div class="dr-header-card">
                    <h1>Synthesis Quality Analysis</h1>
                </div>
                
                {stats_html}
                
                <div class="dr-card">
                    <h2>Confidence Distribution</h2>
                    <div class="dr-chart-container">
                        <canvas id="confidenceChart"></canvas>
                    </div>
                </div>
                
                <div class="synthesis-section">
                    <h2>Synthesized Information</h2>
                    {synthesis_html}
                </div>
            </div>
            
            <script>
                // Confidence distribution bar chart
                const ctx = document.getElementById('confidenceChart').getContext('2d');
                
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: ['High', 'Medium', 'Low'],
                        datasets: [{{
                            label: 'Number of Syntheses',
                            data: [{confidence_counts["high"]}, {confidence_counts["medium"]}, {confidence_counts["low"]}],
                            backgroundColor: ['#27ae60', '#f39c12', '#e74c3c'],
                            borderWidth: 0
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                ticks: {{
                                    stepSize: 1
                                }}
                            }}
                        }},
                        plugins: {{
                            legend: {{
                                display: false
                            }}
                        }}
                    }}
                }});
            </script>
        </body>
        </html>
        """

        return html
