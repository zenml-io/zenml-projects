import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    precision_recall_curve,
    auc,
    roc_auc_score
)
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from typing import Dict, Any, List, Tuple
from typing_extensions import Annotated

from zenml import step
from zenml.materializers.built_in_materializer import BuiltInMaterializer
from zenml.types import HTMLString


def generate_evaluation_html(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    feature_names: List[str],
    feature_importance: np.ndarray,
    metrics: Dict[str, float]
) -> str:
    """Generate HTML visualization of model evaluation results.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        feature_names: Names of selected features
        feature_importance: Importance scores for features
        metrics: Dictionary of evaluation metrics
        
    Returns:
        HTML string with visualizations
    """
    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = px.imshow(
        cm, 
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=["No Subscription", "Subscription"],
        y=["No Subscription", "Subscription"],
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )
    cm_fig.update_layout(width=600, height=500)
    cm_html = cm_fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    )
    roc_fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color='gray'))
    )
    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=600,
        height=500,
        template="plotly_white"
    )
    roc_html = roc_fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    # Create Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_fig = go.Figure()
    pr_fig.add_trace(
        go.Scatter(x=recall, y=precision, mode='lines', name=f'PR (AUC = {metrics["pr_auc"]:.3f})')
    )
    pr_fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        width=600,
        height=500,
        template="plotly_white"
    )
    pr_html = pr_fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    # Create feature importance plot
    indices = np.argsort(feature_importance)[::-1]
    features_sorted = [feature_names[i] for i in indices]
    importance_sorted = feature_importance[indices]
    
    # Only show top 15 features for clarity
    if len(features_sorted) > 15:
        features_sorted = features_sorted[:15]
        importance_sorted = importance_sorted[:15]
    
    fi_fig = px.bar(
        x=importance_sorted,
        y=features_sorted,
        orientation='h',
        title="Feature Importance",
        labels={"x": "Importance", "y": "Feature"},
        color=importance_sorted,
        color_continuous_scale="Viridis"
    )
    fi_fig.update_layout(
        yaxis=dict(autorange="reversed"),
        width=700,
        height=500,
        template="plotly_white"
    )
    fi_html = fi_fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    # Create prediction distribution plot
    pred_df = pd.DataFrame({
        "actual": y_test,
        "predicted_proba": y_pred_proba
    })
    
    dist_fig = px.histogram(
        pred_df,
        x="predicted_proba",
        color="actual",
        nbins=30,
        opacity=0.7,
        barmode="overlay",
        title="Distribution of Prediction Probabilities",
        labels={"predicted_proba": "Predicted Probability", "actual": "Actual Subscription"},
        color_discrete_map={0: "#636EFA", 1: "#EF553B"}
    )
    dist_fig.update_layout(
        width=700,
        height=500,
        template="plotly_white"
    )
    dist_html = dist_fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    # Combine HTML elements
    html_parts = []
    html_parts.append("""
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f9f9f9;
            }
            .dashboard {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px 30px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .section {
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            h1, h2, h3 {
                color: #333;
            }
            .metrics {
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin-bottom: 30px;
            }
            .metric-card {
                flex: 1;
                min-width: 200px;
                margin: 10px;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                text-align: center;
            }
            .metric-value {
                font-size: 28px;
                font-weight: bold;
                margin: 10px 0;
            }
            .row {
                display: flex;
                flex-wrap: wrap;
                margin: 0 -15px;
            }
            .column {
                flex: 50%;
                padding: 0 15px;
                box-sizing: border-box;
            }
            @media (max-width: 800px) {
                .column {
                    flex: 100%;
                }
            }
            .plot-container {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <div class="section">
                <h1>Bank Subscription Prediction Model Evaluation</h1>
                <p>Evaluation of XGBoost model performance on the test set.</p>
            </div>
    """)
    
    # Add metrics cards
    html_parts.append("""
            <div class="section">
                <h2>Model Performance Metrics</h2>
                <div class="metrics">
    """)
    
    # Accuracy card
    html_parts.append(f"""
                    <div class="metric-card" style="background-color: #f0f8ff;">
                        <h3>Accuracy</h3>
                        <div class="metric-value">{metrics["accuracy"]:.3f}</div>
                        <p>Proportion of correctly classified instances</p>
                    </div>
    """)
    
    # ROC AUC card
    html_parts.append(f"""
                    <div class="metric-card" style="background-color: #fff8f0;">
                        <h3>ROC AUC</h3>
                        <div class="metric-value">{metrics["roc_auc"]:.3f}</div>
                        <p>Area under the ROC curve</p>
                    </div>
    """)
    
    # PR AUC card
    html_parts.append(f"""
                    <div class="metric-card" style="background-color: #f0fff8;">
                        <h3>PR AUC</h3>
                        <div class="metric-value">{metrics["pr_auc"]:.3f}</div>
                        <p>Area under the Precision-Recall curve</p>
                    </div>
    """)
    
    # Close metrics section
    html_parts.append("""
                </div>
            </div>
    """)
    
    # Add plots in two columns
    html_parts.append("""
            <div class="section">
                <h2>Visualizations</h2>
                <div class="row">
                    <div class="column">
                        <div class="plot-container">
    """)
    
    # Add confusion matrix
    html_parts.append(f"""
                            <h3>Confusion Matrix</h3>
                            {cm_html}
    """)
    
    # Add ROC curve
    html_parts.append(f"""
                        </div>
                        <div class="plot-container">
                            <h3>ROC Curve</h3>
                            {roc_html}
                        </div>
                    </div>
                    <div class="column">
                        <div class="plot-container">
                            <h3>Precision-Recall Curve</h3>
                            {pr_html}
    """)
    
    # Add prediction distribution
    html_parts.append(f"""
                        </div>
                        <div class="plot-container">
                            <h3>Prediction Distribution</h3>
                            {dist_html}
                        </div>
                    </div>
                </div>
    """)
    
    # Add feature importance
    html_parts.append(f"""
                <h2>Feature Importance</h2>
                <div class="plot-container">
                    {fi_html}
                </div>
    """)
    
    # Close the HTML document
    html_parts.append("""
            </div>
        </div>
    </body>
    </html>
    """)
    
    # Combine all HTML parts
    return "".join(html_parts) 

@step
def evaluate_model(
    model: xgb.XGBClassifier,
    feature_selector: SelectFromModel,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[Dict[str, float], "metrics"],
    Annotated[HTMLString, "evaluation_visualization"]
]:
    """Evaluates the trained model on the test set.

    Args:
        model: Trained XGBoost model
        feature_selector: Feature selector used in training
        X_test: Test features
        y_test: Test targets

    Returns:
        metrics: Dictionary of evaluation metrics
        evaluation_visualization: HTML visualization of evaluation results
    """
    # Apply feature selection to test data
    X_test_selected = feature_selector.transform(X_test)
    
    # Get predictions
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Feature importance
    feature_names = X_test.columns[feature_selector.get_support()]
    importance = model.feature_importances_
    
    # Store metrics
    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    
    # Generate HTML visualization
    html_content = generate_evaluation_html(
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        feature_names=feature_names,
        feature_importance=importance,
        metrics=metrics
    )
    
    return metrics, HTMLString(html_content)
