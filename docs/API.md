# API Reference

## Overview

RESTful API for credit risk assessment predictions. Built with Flask/FastAPI.

## Endpoints

### 1. Health Check

```http
GET /health
```

**Response** (200 OK):
```json
{
  "status": "healthy",
  "model_version": "1.0.0",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### 2. Predict Single Application

```http
POST /predict
Content-Type: application/json
```

**Request Body**:
```json
{
  "applicant_data": {
    "age": 35,
    "annual_income": 75000,
    "credit_score": 720,
    "employment_years": 8,
    "num_accounts": 5,
    "credit_utilization": 0.45,
    "num_inquiries": 2,
    "payment_history": "excellent",
    "loan_purpose": "auto",
    "existing_debt": 15000
  }
}
```

**Response** (200 OK):
```json
{
  "prediction": 0,
  "risk_probability": 0.15,
  "risk_level": "low",
  "confidence": 0.95,
  "model_version": "1.0.0",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### 3. Batch Predictions

```http
POST /predict_batch
Content-Type: application/json
```

**Request Body**:
```json
{
  "applicants": [
    {"age": 35, "annual_income": 75000, ...},
    {"age": 42, "annual_income": 95000, ...}
  ]
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "applicant_id": 0,
      "prediction": 0,
      "risk_probability": 0.15,
      "risk_level": "low"
    },
    {
      "applicant_id": 1,
      "prediction": 0,
      "risk_probability": 0.08,
      "risk_level": "low"
    }
  ],
  "processed": 2,
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### 4. Get Prediction Explanation (SHAP)

```http
POST /explain
Content-Type: application/json
```

**Request Body**:
```json
{
  "applicant_data": {
    "age": 35,
    "annual_income": 75000,
    ...
  }
}
```

**Response** (200 OK):
```json
{
  "base_value": 0.12,
  "prediction": 0.15,
  "shap_values": {
    "annual_income": 0.02,
    "credit_score": -0.01,
    "num_inquiries": 0.03,
    "payment_history": -0.01,
    ...
  },
  "top_factors": [
    {"feature": "num_inquiries", "impact": 0.03, "direction": "increases_risk"},
    {"feature": "annual_income", "impact": 0.02, "direction": "increases_risk"}
  ]
}
```

### 5. Model Metrics

```http
GET /metrics
```

**Response** (200 OK):
```json
{
  "model_version": "1.0.0",
  "accuracy": 0.87,
  "precision": 0.82,
  "recall": 0.78,
  "f1_score": 0.80,
  "roc_auc": 0.88,
  "training_data": {
    "samples": 307511,
    "features": 85,
    "train_test_split": "80-20"
  },
  "last_update": "2024-01-15T09:00:00Z"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid input",
  "message": "Missing required field: annual_income",
  "code": "INVALID_INPUT"
}
```

### 500 Internal Server Error
```json
{
  "error": "Server error",
  "message": "Failed to load model",
  "code": "MODEL_LOAD_ERROR"
}
```

## Authentication

Add to future versions:
```
Authorization: Bearer <API_KEY>
```

## Rate Limiting

- Standard: 100 requests/minute
- Batch: 10 requests/minute
- Burst: 500 requests/hour

## Python Client Example

```python
import requests

api_url = "http://localhost:5000"

# Single prediction
response = requests.post(
    f"{api_url}/predict",
    json={
        "applicant_data": {
            "age": 35,
            "annual_income": 75000,
            "credit_score": 720,
            ...
        }
    }
)

result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Risk Probability: {result['risk_probability']:.2%}")
```

## Integration Examples

### JavaScript/Node.js
```javascript
const response = await fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({applicant_data: {...}})
});
const result = await response.json();
```

### cURL
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "applicant_data": {
      "age": 35,
      "annual_income": 75000,
      ...
    }
  }'
```

