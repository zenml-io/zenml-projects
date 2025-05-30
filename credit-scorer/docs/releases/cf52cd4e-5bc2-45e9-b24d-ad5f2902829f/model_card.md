# Credit Risk Assessment Model

## Model Details

**Model ID:** bff4d94d
**Version:** 2025-05-22
**Description:** This model assesses credit risk for loan applications
**Type:** LGBMClassifier
**Framework:** LightGBM

## Intended Use

This model is designed to assist financial institutions in assessing credit risk for loan applicants. It predicts the probability of loan default based on applicant financial and demographic data. The primary use case is to support human decision-makers in loan approval processes.

## Performance Metrics

### Model Quality (Threshold-Independent)

| Metric | Value | Description |
|--------|-------|-------------|

### Performance at Standard Threshold (0.5)

| Metric | Value |
|--------|---------|
| accuracy | 0.8399 |
| precision | 0.2246 |
| recall | 0.4203 |
| f1_score | 0.2927 |

### Performance at Optimal Threshold (0.63)

| Metric | Value |
|--------|---------|

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

| Metric | Value |
|--------|-------|
| selection_rate_disparity | 0.1143 |

#### Age Years Groups

| Metric | Value |
|--------|-------|
| selection_rate_disparity | 0.2273 |

#### Name Education Type Groups

| Metric | Value |
|--------|-------|
| selection_rate_disparity | 0.1483 |

#### Name Family Status

| Metric | Value |
|--------|-------|
| selection_rate_disparity | 0.1387 |

#### Name Housing Type

| Metric | Value |
|--------|-------|
| selection_rate_disparity | 0.1547 |

## Limitations

- Model performance may degrade when faced with economic conditions significantly different from the training period
- Limited validation on certain demographic groups due to data availability
- Does not incorporate alternative credit data (utility payments, rent history)
- May not generalize well to loan types or amounts significantly different from training distribution

## Risk Management

The overall risk score for this model is 0.65 on a scale of 0-1 (lower is better).
This model is subject to continuous monitoring for data drift and performance degradation.
Human oversight is required for all decisions made with assistance from this model.

