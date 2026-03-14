# ☁️ Serverless Insurance Risk Scoring API

> **AWS Solutions Architect Demo Project**
> A production-grade, fully serverless REST API that scores insurance risk in real time using machine learning — deployed entirely on AWS managed services with Infrastructure as Code.

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────────────────────────┐
                          │              AWS Cloud (us-east-1)               │
                          │                                                   │
  Client / Postman  ────► │  API Gateway  ──►  Lambda Function               │
                          │  (REST API)         (Python 3.11)                │
                          │                      │                           │
                          │            ┌─────────┼──────────┐               │
                          │            ▼         ▼          ▼               │
                          │         S3 Bucket  DynamoDB   SNS Topic         │
                          │         (Models)   (Audit Log) (Alerts)         │
                          │            │                    │               │
                          │            └── Lifecycle ──►    └──► SES Email  │
                          │                (Glacier)                        │
                          │                                                   │
                          │         CloudWatch (Logs + Alarms)               │
                          │         IAM (Least-privilege roles)              │
                          └─────────────────────────────────────────────────┘
```

---

## ✅ AWS Services Demonstrated

| Service | Purpose | SA Exam Domain |
|---------|---------|----------------|
| **API Gateway** | REST API with throttling & staging | Design Resilient Architectures |
| **Lambda** | Serverless compute with warm-start caching | Design Cost-Optimized Architectures |
| **S3** | Model storage with lifecycle → Glacier | Design Cost-Optimized Architectures |
| **DynamoDB** | On-demand audit log with TTL & encryption | Design Secure Architectures |
| **SNS + SES** | High-risk alert pipeline | Design Event-Driven Architectures |
| **IAM** | Least-privilege execution role | Design Secure Architectures |
| **CloudWatch** | Alarms, logs, custom metrics | Design Resilient Architectures |
| **CloudFormation** | Full IaC — one-command deploy | All Domains |

---

## 📁 Project Structure

```
serverless-risk-api/
│
├── lambda/
│   ├── handler.py              # Lambda function — risk scoring logic
│   └── requirements.txt        # Runtime dependencies
│
├── model/
│   └── train.py                # Train model + upload to S3
│
├── cloudformation/
│   └── template.yaml           # Full IaC stack definition
│
├── scripts/
│   └── deploy.sh               # One-command deployment script
│
├── tests/
│   └── test_handler.py         # Full test suite (pytest)
│
├── requirements-dev.txt        # Local dev + testing dependencies
└── README.md
```

---

## 🔌 API Reference

### `POST /score-risk`
Score an insurance applicant's risk profile.

**Request Body:**
```json
{
  "age": 45,
  "credit_score": 680,
  "claims_history": 2,
  "policy_type": "auto",
  "years_insured": 5,
  "annual_mileage": 12000,
  "property_value": 0,
  "prior_cancellation": false
}
```

| Field | Type | Required | Range |
|-------|------|----------|-------|
| `age` | number | ✅ | 18–100 |
| `credit_score` | number | ✅ | 300–850 |
| `claims_history` | number | ✅ | 0–20 |
| `policy_type` | string | ✅ | auto, home, life, health, commercial |
| `years_insured` | number | ❌ | 0–40 |
| `annual_mileage` | number | ❌ | 0–50000 |
| `property_value` | number | ❌ | 0+ |
| `prior_cancellation` | boolean | ❌ | true/false |

**Response (200 OK):**
```json
{
  "request_id": "a3f9b2c1-...",
  "risk_score": 0.6234,
  "risk_tier": {
    "tier": "ELEVATED",
    "label": "Substandard Risk",
    "premium_adjustment": "+15%",
    "recommendation": "Approve with surcharge — review in 12 months",
    "color": "orange"
  },
  "model_version": "1.0.0",
  "scored_at": "2025-03-14T14:22:00Z"
}
```

**Risk Tiers:**

| Score Range | Tier | Premium Adjustment |
|-------------|------|--------------------|
| 0.00 – 0.24 | 🟢 LOW | -10% |
| 0.25 – 0.49 | 🔵 MODERATE | 0% |
| 0.50 – 0.74 | 🟠 ELEVATED | +15% |
| 0.75 – 1.00 | 🔴 HIGH | +30% |

### `GET /health`
Health check endpoint.

```json
{"status": "healthy", "version": "1.0.0"}
```

---

## 🚀 Deployment Guide

### Prerequisites
- AWS CLI configured (`aws configure`)
- Python 3.11+
- Permissions: CloudFormation, Lambda, S3, DynamoDB, API Gateway, SNS, IAM

### 1. Clone & install dependencies
```bash
git clone https://github.com/your-username/serverless-risk-api.git
cd serverless-risk-api
pip install -r requirements-dev.txt
```

### 2. Deploy everything (one command)
```bash
chmod +x scripts/deploy.sh
./scripts/deploy.sh dev your-aws-profile
```

This script will:
1. ✅ Validate the CloudFormation template
2. ✅ Deploy the full AWS stack
3. ✅ Train the ML model on synthetic data
4. ✅ Upload the model artifact to S3
5. ✅ Package and deploy the Lambda function
6. ✅ Run a smoke test against the live API

### 3. Test the live API
```bash
# Get your API URL from deployment output, then:
curl -X POST https://<api-id>.execute-api.us-east-1.amazonaws.com/dev/score-risk \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "credit_score": 620,
    "claims_history": 3,
    "policy_type": "auto",
    "prior_cancellation": true
  }'
```

### 4. Teardown (avoid charges)
```bash
aws cloudformation delete-stack --stack-name risk-scoring-api-dev
```

---

## 🧪 Running Tests Locally

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=lambda --cov-report=term-missing
```

**Test Coverage:**
- ✅ Input validation (valid + 8 edge cases)
- ✅ Risk tier assignment + boundary conditions
- ✅ Feature engineering + encoding
- ✅ Lambda handler (success, errors, alerts)
- ✅ CORS headers
- ✅ Health check

---

## 🔐 Security Design Decisions

| Decision | Rationale |
|----------|-----------|
| **IAM least-privilege role** | Lambda only has S3 read, DynamoDB write, SNS publish — nothing more |
| **S3 bucket — all public access blocked** | Model artifacts are private; accessed only by Lambda via IAM |
| **DynamoDB encryption at rest** | SSE enabled by default for all stored scoring records |
| **DynamoDB TTL** | Records auto-expire after 90 days — minimizes PII exposure |
| **API Gateway throttling** | 50 req/sec rate limit + 100 burst to prevent abuse |
| **CloudWatch alarms** | Automated alerting on errors and latency spikes |

---

## 💰 Cost Estimate (Dev Tier)

| Service | Free Tier | Est. Cost (1K req/day) |
|---------|-----------|------------------------|
| Lambda | 1M req/month free | ~$0.00 |
| API Gateway | 1M req/month free | ~$0.00 |
| DynamoDB | 25GB + 200M req free | ~$0.00 |
| S3 | 5GB free | ~$0.01 |
| CloudWatch | Basic metrics free | ~$0.00 |
| SNS | 1M notifications free | ~$0.00 |
| **Total** | | **~$0.01/month** |

---

## 🧠 ML Model Details

The ensemble model is trained on insurance risk features using:

| Model | Role |
|-------|------|
| Random Forest | Handles non-linear patterns |
| Gradient Boosting | High accuracy on structured data |
| Logistic Regression | Interpretable baseline |

Best model selected by ROC-AUC score at training time.

**Features used:**
- Age, credit score, claims history
- Policy type, years insured
- Annual mileage (auto policies)
- Property value (home policies)
- Prior cancellation flag

---

## 📊 What This Demonstrates for Solutions Architect

- **Multi-service integration** across 8+ AWS services
- **Serverless architecture** with no server management
- **Infrastructure as Code** via CloudFormation (one-command deploy + destroy)
- **Event-driven design** (SNS alerts triggered by business logic)
- **Cost optimization** (S3 lifecycle, DynamoDB on-demand, Lambda pay-per-use)
- **Security** (IAM least-privilege, encryption, public access blocks)
- **Resilience** (CloudWatch alarms, structured error handling, health checks)
- **Observability** (structured logging, custom metrics, audit trail)




