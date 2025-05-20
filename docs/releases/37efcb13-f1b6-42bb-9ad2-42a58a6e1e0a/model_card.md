# Credit Risk Assessment Model

## Model Details

**Model ID:** bb94f3d9
**Version:** 2025-05-19
**Description:** This model assesses credit risk for loan applications
**Type:** Gradient Boosting Classifier
**Framework:** scikit-learn

## Intended Use

This model is designed to assist financial institutions in assessing credit risk for loan applicants. It predicts the probability of loan default based on applicant financial and demographic data. The primary use case is to support human decision-makers in loan approval processes.

## Performance Metrics

| Metric   | Value  |
| -------- | ------ |
| accuracy | 0.9189 |
| auc      | 0.7500 |

### Decision Thresholds

Model outputs a probability score (0-1). Recommended thresholds:

- Low risk: 0.0-0.3
- Medium risk: 0.3-0.6
- High risk: 0.6-1.0

## Fairness Considerations

The model has been evaluated for fairness across different demographic groups. We implement several measures to mitigate potential bias, including:

- Protected attributes are not directly used as features
- Fairness metrics are evaluated across different demographic groups
- Post-processing techniques applied to reduce disparate impact

### Fairness Metrics

#### Code Gender

| Metric                   | Value  |
| ------------------------ | ------ |
| selection_rate_disparity | 0.0034 |

#### Num Age Years

| Metric                   | Value  |
| ------------------------ | ------ |
| selection_rate_disparity | 1.0000 |

#### Name Education Type

| Metric                   | Value  |
| ------------------------ | ------ |
| selection_rate_disparity | 0.0217 |

#### Name Family Status

| Metric                   | Value  |
| ------------------------ | ------ |
| selection_rate_disparity | 0.0045 |

#### Name Housing Type

| Metric                   | Value  |
| ------------------------ | ------ |
| selection_rate_disparity | 0.0175 |

## Limitations

- Model performance may degrade when faced with economic conditions significantly different from the training period
- Limited validation on certain demographic groups due to data availability
- Does not incorporate alternative credit data (utility payments, rent history)
- May not generalize well to loan types or amounts significantly different from training distribution

## Risk Management

The overall risk score for this model is 0.53 on a scale of 0-1 (lower is better).
This model is subject to continuous monitoring for data drift and performance degradation.
Human oversight is required for all decisions made with assistance from this model.

## Contact Information

For questions or concerns about this model, please contact compliance@example.com
