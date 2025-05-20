# Credit Scoring API Guide

## Overview

The Credit Scoring API is a EU AI Act compliant service that provides credit risk assessment for loan applications. The API is deployed using Modal and FastAPI, offering a serverless, scalable solution with automatic scaling capabilities.

## Base URL

The API is deployed at: `https://{workspace_name}-{environment}--{app_name}.modal.run`

## Authentication

Currently, the API does not require authentication. However, it is recommended to implement authentication in production environments.

## Endpoints

### 1. Root Endpoint

```http
GET /
```

Returns basic API information and available endpoints.

**Response:**

```json
{
  "message": "Credit Scoring API - EU AI Act Compliant",
  "endpoints": {
    "root": "https://example.modal.run",
    "health": "https://example.modal.run/health",
    "predict": "https://example.modal.run/predict",
    "monitor": "https://example.modal.run/monitor",
    "incident": "https://example.modal.run/incident"
  },
  "model": "credit-scoring-model",
  "timestamp": "2024-03-20T10:00:00Z"
}
```

### 2. Health Check

```http
GET /health
```

Checks the health status of the API.

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2024-03-20T10:00:00Z"
}
```

### 3. Prediction Endpoint

```http
POST /predict
```

Makes credit risk predictions based on applicant data.

**Request Body:**

```json
{
  "age": 35,
  "income": 50000,
  "employment_length": 5,
  "credit_history_length": 10,
  "debt_to_income_ratio": 0.3,
  "payment_history": 0.95,
  "credit_score": 720
}
```

**Response:**

```json
{
  "probabilities": [0.75],
  "model_version": "a1b2c3d4",
  "timestamp": "2024-03-20T10:00:00Z",
  "risk_assessment": {
    "risk_score": 0.75,
    "risk_level": "high"
  }
}
```

### 4. Monitoring Endpoint

```http
GET /monitor
```

Checks for data drift and model performance issues.

**Response:**

```json
{
  "timestamp": "2024-03-20T10:00:00Z",
  "drift_detected": false,
  "status": "healthy",
  "message": "No significant drift detected"
}
```

### 5. Incident Reporting

```http
POST /incident
```

Reports incidents or issues with the system.

**Request Body:**

```json
{
  "severity": "medium",
  "description": "Unusual pattern detected in predictions",
  "data": {
    "affected_predictions": 10,
    "timeframe": "2024-03-20T09:00:00Z to 2024-03-20T10:00:00Z"
  }
}
```

**Response:**

```json
{
  "status": "received",
  "incident_id": "inc_123456",
  "message": "Incident report received and logged",
  "timestamp": "2024-03-20T10:00:00Z"
}
```

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad Request
- 500: Internal Server Error

Error responses include a detail message:

```json
{
  "detail": "Error message describing the issue"
}
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- 100 requests per minute per IP address
- 1000 requests per hour per IP address

## Best Practices

1. Always check the health endpoint before making predictions
2. Implement proper error handling in your client code
3. Monitor the response timestamps for potential delays
4. Report any unusual patterns or issues through the incident endpoint
5. Keep track of the model version in responses for audit purposes

## Compliance

This API is designed to comply with the EU AI Act requirements, including:

- Article 9: Risk Management
- Article 10: Data Governance
- Article 11: Technical Documentation
- Article 13: Transparency
- Article 14: Human Oversight
- Article 15: Accuracy and Robustness
- Article 17: Post-market Monitoring
- Article 18: Incident Reporting
