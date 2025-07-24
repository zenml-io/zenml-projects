"""
Evaluation step for assessing pipeline quality and performance metrics.
"""

from datetime import datetime
from typing import List, Tuple

from typing_extensions import Annotated
from zenml import step
from zenml.logger import get_logger
from zenml.types import HTMLString

from ..utils.models import (
    DeliveryResult,
    EvaluationMetrics,
    ProcessedData,
    RawConversationData,
)

logger = get_logger(__name__)


class PipelineEvaluator:
    """Evaluates the quality and performance of the pipeline."""

    def evaluate_summary_quality(self, processed_data: ProcessedData) -> float:
        """Evaluate the quality of generated summaries."""

        if not processed_data.summaries:
            return 0.0

        total_score = 0.0

        for summary in processed_data.summaries:
            score = 0.0

            # Length check (reasonable summary length)
            word_count = summary.word_count
            if 50 <= word_count <= 500:
                score += 0.3
            elif word_count > 20:
                score += 0.1

            # Content quality (basic checks)
            if len(summary.key_points) > 0:
                score += 0.2

            if len(summary.participants) > 0:
                score += 0.2

            if len(summary.topics) > 0:
                score += 0.1

            # Use the agent's confidence score
            score += summary.confidence_score * 0.2

            total_score += min(score, 1.0)

        return total_score / len(processed_data.summaries)

    def evaluate_task_extraction_accuracy(
        self, processed_data: ProcessedData
    ) -> float:
        """Evaluate the accuracy of task extraction."""

        if not processed_data.tasks:
            return 0.5  # Neutral score if no tasks found

        total_score = 0.0

        for task in processed_data.tasks:
            score = 0.0

            # Title quality
            if len(task.title) >= 5:
                score += 0.2

            # Description quality
            if len(task.description) >= 10:
                score += 0.3

            # Priority assignment
            if task.priority in ["high", "medium", "low"]:
                score += 0.1

            # Confidence score
            score += task.confidence_score * 0.4

            total_score += min(score, 1.0)

        return total_score / len(processed_data.tasks)

    def calculate_delivery_success_rate(
        self, delivery_results: List[DeliveryResult]
    ) -> float:
        """Calculate the success rate of output delivery."""

        if not delivery_results:
            return 0.0

        successful_deliveries = sum(
            1 for result in delivery_results if result.success
        )
        return successful_deliveries / len(delivery_results)

    def estimate_cost(self, processed_data: ProcessedData) -> float:
        """Estimate the cost of LLM operations."""

        # Rough cost estimation for Gemini 2.5 Flash
        # Input: ~$0.30 per 1M tokens, Output: ~$2.50 per 1M tokens

        total_tokens = processed_data.llm_usage_stats.get("total_tokens", 0)
        api_calls = processed_data.llm_usage_stats.get("api_calls", 0)

        # Estimate input tokens (conversation data + prompts)
        estimated_input_tokens = total_tokens * 3  # Rough multiplier for input
        estimated_output_tokens = total_tokens

        input_cost = (estimated_input_tokens / 1_000_000) * 0.30
        output_cost = (estimated_output_tokens / 1_000_000) * 2.50

        return input_cost + output_cost

    def calculate_processing_time(
        self, processed_data: ProcessedData
    ) -> float:
        """Calculate the total processing time."""

        start_time_str = processed_data.processing_metadata.get("start_time")
        end_time_str = processed_data.processing_metadata.get("end_time")

        if start_time_str and end_time_str:
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.fromisoformat(end_time_str)
            return (end_time - start_time).total_seconds()

        return processed_data.processing_metadata.get(
            "processing_time_seconds", 0.0
        )


def _create_evaluation_dashboard(
    metrics: EvaluationMetrics,
    processed_data: ProcessedData,
    raw_conversations: RawConversationData,
    delivery_results: List[DeliveryResult],
) -> HTMLString:
    """Create an HTML dashboard for evaluation metrics."""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pipeline Evaluation Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric {{ text-align: center; margin: 10px 0; }}
            .metric-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
            .metric-label {{ color: #666; text-transform: uppercase; font-size: 0.9em; }}
            .progress-bar {{ width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; }}
            .progress-fill {{ height: 100%; background: linear-gradient(90deg, #4CAF50, #2196F3); transition: width 0.3s; }}
            .summary-card {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
            h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
            .status-good {{ color: #4CAF50; }}
            .status-warning {{ color: #FF9800; }}
            .status-error {{ color: #f44336; }}
        </style>
    </head>
    <body>
        <h1>🚀 Pipeline Evaluation Dashboard</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="dashboard">
            <div class="card">
                <h2>📊 Quality Metrics</h2>
                <div class="metric">
                    <div class="metric-value status-{'good' if metrics.summary_quality_score > 0.7 else 'warning' if metrics.summary_quality_score > 0.4 else 'error'}">{metrics.summary_quality_score:.1%}</div>
                    <div class="metric-label">Summary Quality</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {metrics.summary_quality_score * 100}%"></div>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-value status-{'good' if metrics.task_extraction_accuracy > 0.7 else 'warning' if metrics.task_extraction_accuracy > 0.4 else 'error'}">{metrics.task_extraction_accuracy:.1%}</div>
                    <div class="metric-label">Task Accuracy</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {metrics.task_extraction_accuracy * 100}%"></div>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-value status-{'good' if metrics.delivery_success_rate > 0.7 else 'warning' if metrics.delivery_success_rate > 0.4 else 'error'}">{metrics.delivery_success_rate:.1%}</div>
                    <div class="metric-label">Delivery Success</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {metrics.delivery_success_rate * 100}%"></div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>⚡ Performance Metrics</h2>
                <div class="metric">
                    <div class="metric-value">{metrics.processing_time_seconds:.1f}s</div>
                    <div class="metric-label">Processing Time</div>
                </div>
                <div class="metric">
                    <div class="metric-value">${metrics.cost_estimate:.4f}</div>
                    <div class="metric-label">Estimated Cost</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.token_usage.get('total_tokens', 0):,}</div>
                    <div class="metric-label">Total Tokens</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{metrics.token_usage.get('api_calls', 0)}</div>
                    <div class="metric-label">API Calls</div>
                </div>
            </div>
            
            <div class="card">
                <h2>📝 Content Summary</h2>
                <div class="metric">
                    <div class="metric-value">{len(processed_data.summaries)}</div>
                    <div class="metric-label">Summaries Generated</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(processed_data.tasks)}</div>
                    <div class="metric-label">Tasks Extracted</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(raw_conversations.conversations)}</div>
                    <div class="metric-label">Conversations Processed</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{sum(conv.total_messages for conv in raw_conversations.conversations)}</div>
                    <div class="metric-label">Total Messages</div>
                </div>
            </div>
            
            <div class="card">
                <h2>🎯 Generated Summaries</h2>
                {''.join([f'''
                <div class="summary-card">
                    <strong>Title:</strong> {summary.title}<br>
                    <strong>Topics:</strong> {', '.join(summary.topics[:3])}{'...' if len(summary.topics) > 3 else ''}<br>
                    <strong>Key Points:</strong> {len(summary.key_points)}<br>
                    <strong>Participants:</strong> {len(summary.participants)}<br>
                    <strong>Confidence:</strong> {summary.confidence_score:.1%}
                </div>
                ''' for summary in processed_data.summaries[:5]])}
                {f'<p><em>... and {len(processed_data.summaries) - 5} more summaries</em></p>' if len(processed_data.summaries) > 5 else ''}
            </div>
            
            <div class="card">
                <h2>✅ Extracted Tasks</h2>
                {''.join([f'''
                <div class="summary-card">
                    <strong>Title:</strong> {task.title}<br>
                    <strong>Priority:</strong> <span class="status-{'good' if task.priority == 'high' else 'warning' if task.priority == 'medium' else 'error'}">{task.priority.upper()}</span><br>
                    <strong>Assigned to:</strong> {task.assignee or 'Unassigned'}<br>
                    <strong>Confidence:</strong> {task.confidence_score:.1%}
                </div>
                ''' for task in processed_data.tasks[:5]])}
                {f'<p><em>... and {len(processed_data.tasks) - 5} more tasks</em></p>' if len(processed_data.tasks) > 5 else ''}
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLString(html_content)


@step
def evaluation_step(
    summaries_and_tasks: ProcessedData,
    raw_conversations: RawConversationData,
    delivery_results: List[DeliveryResult],
) -> Tuple[
    Annotated[EvaluationMetrics, "evaluation_metrics"],
    Annotated[HTMLString, "evaluation_dashboard"],
]:
    """
    Evaluate the pipeline performance and quality metrics.

    Args:
        summaries_and_tasks: Processed data from LangGraph agents
        raw_conversations: Original conversation data for reference
        delivery_results: Results from output distribution

    Returns:
        Tuple[EvaluationMetrics, HTMLString]: Comprehensive evaluation metrics and HTML dashboard
    """

    logger.info("Starting pipeline evaluation")

    evaluator = PipelineEvaluator()

    # Evaluate summary quality
    summary_quality_score = evaluator.evaluate_summary_quality(
        summaries_and_tasks
    )
    logger.info(f"Summary quality score: {summary_quality_score:.3f}")

    # Evaluate task extraction accuracy
    task_extraction_accuracy = evaluator.evaluate_task_extraction_accuracy(
        summaries_and_tasks
    )
    logger.info(f"Task extraction accuracy: {task_extraction_accuracy:.3f}")

    # Calculate delivery success rate
    delivery_success_rate = evaluator.calculate_delivery_success_rate(
        delivery_results
    )
    logger.info(f"Delivery success rate: {delivery_success_rate:.3f}")

    # Calculate processing time
    processing_time = evaluator.calculate_processing_time(summaries_and_tasks)
    logger.info(f"Processing time: {processing_time:.2f} seconds")

    # Estimate cost
    cost_estimate = evaluator.estimate_cost(summaries_and_tasks)
    logger.info(f"Estimated cost: ${cost_estimate:.4f}")

    # Extract token usage
    token_usage = summaries_and_tasks.llm_usage_stats

    # Create evaluation metrics
    metrics = EvaluationMetrics(
        summary_quality_score=summary_quality_score,
        task_extraction_accuracy=task_extraction_accuracy,
        processing_time_seconds=processing_time,
        token_usage=token_usage,
        cost_estimate=cost_estimate,
        delivery_success_rate=delivery_success_rate,
        human_feedback_score=None,  # Could be added later with human evaluation
    )

    # Log summary
    logger.info("Pipeline evaluation complete:")
    logger.info(f"  - Summary Quality: {summary_quality_score:.3f}")
    logger.info(f"  - Task Accuracy: {task_extraction_accuracy:.3f}")
    logger.info(f"  - Delivery Success: {delivery_success_rate:.3f}")
    logger.info(f"  - Processing Time: {processing_time:.2f}s")
    logger.info(f"  - Cost: ${cost_estimate:.4f}")
    logger.info(f"  - API Calls: {token_usage.get('api_calls', 0)}")
    logger.info(f"  - Total Tokens: {token_usage.get('total_tokens', 0)}")

    # Create HTML visualization
    html_viz = _create_evaluation_dashboard(
        metrics, summaries_and_tasks, raw_conversations, delivery_results
    )

    return metrics, html_viz
