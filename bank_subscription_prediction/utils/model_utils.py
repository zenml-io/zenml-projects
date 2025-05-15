"""Utility functions for model operations."""

import pandas as pd
import numpy as np


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Calculate the scale_pos_weight parameter for imbalanced classification.
    
    Args:
        y: Target variable series (binary: 0/1)
        
    Returns:
        The ratio of negative to positive samples
    """
    # Count occurrences of each class
    class_counts = y.value_counts()
    
    # For binary classification (0/1)
    if 0 in class_counts and 1 in class_counts:
        neg_count = class_counts[0]
        pos_count = class_counts[1]
        return neg_count / pos_count
    else:
        print("Warning: Could not calculate scale_pos_weight. Using default value 1.")
        return 1.0


def get_feature_importance(model, feature_names):
    """Get feature importance from a trained XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature names and their importance scores
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    feature_importance_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importance[indices]
    })
    
    return feature_importance_df 