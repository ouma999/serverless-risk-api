"""
Serverless Insurance Risk Scoring API
Lambda Handler — loads ML model from S3 and returns risk scores
"""

import json
import os
import boto3
import pickle
import numpy as np
import logging
from datetime import datetime, timezone
from decimal import Decimal

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS clients (initialized outside handler for Lambda warm-start reuse)
s3_client = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sns_client = boto3.client("sns")

# Environment variables (set in CloudFormation)
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "risk-scoring-models")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/risk_model.pkl")
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "RiskScoreResults")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "")
HIGH_RISK_THRESHOLD = float(os.environ.get("HIGH_RISK_THRESHOLD", "0.75"))

# Global model cache (persists across warm Lambda invocations)
_model = None
_scaler = None


def load_model():
    """Load ML model from S3 — cached after first load."""
    global _model, _scaler
    if _model is None:
        logger.info(f"Loading model from s3://{MODEL_BUCKET}/{MODEL_KEY}")
        response = s3_client.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
        model_data = pickle.loads(response["Body"].read())
        _model = model_data["model"]
        _scaler = model_data.get("scaler")
        logger.info("Model loaded successfully")
    return _model, _scaler


def validate_input(body: dict) -> tuple[bool, str]:
    """Validate incoming request payload."""
    required_fields = {
        "age": (18, 100),
        "credit_score": (300, 850),
        "claims_history": (0, 20),
    }
    valid_policy_types = ["auto", "home", "life", "health", "commercial"]

    for field, (min_val, max_val) in required_fields.items():
        if field not in body:
            return False, f"Missing required field: '{field}'"
        try:
            val = float(body[field])
            if not (min_val <= val <= max_val):
                return False, f"'{field}' must be between {min_val} and {max_val}"
        except (ValueError, TypeError):
            return False, f"'{field}' must be a number"

    if "policy_type" not in body:
        return False, "Missing required field: 'policy_type'"
    if body["policy_type"].lower() not in valid_policy_types:
        return False, f"'policy_type' must be one of: {valid_policy_types}"

    return True, ""


def build_features(body: dict) -> np.ndarray:
    """Transform raw input into model feature vector."""
    policy_type_map = {"auto": 0, "home": 1, "life": 2, "health": 3, "commercial": 4}

    features = np.array([
        float(body["age"]),
        float(body["credit_score"]),
        float(body["claims_history"]),
        policy_type_map.get(body.get("policy_type", "auto").lower(), 0),
        float(body.get("years_insured", 3)),
        float(body.get("annual_mileage", 12000)) if body.get("policy_type") == "auto" else 0,
        float(body.get("property_value", 0)),
        1 if body.get("prior_cancellation", False) else 0,
    ]).reshape(1, -1)

    return features


def get_risk_tier(score: float) -> dict:
    """Map numeric risk score to human-readable tier and recommendations."""
    if score < 0.25:
        return {
            "tier": "LOW",
            "label": "Preferred Risk",
            "premium_adjustment": "-10%",
            "recommendation": "Approve — offer preferred rate",
            "color": "green"
        }
    elif score < 0.50:
        return {
            "tier": "MODERATE",
            "label": "Standard Risk",
            "premium_adjustment": "0%",
            "recommendation": "Approve — standard rate",
            "color": "blue"
        }
    elif score < 0.75:
        return {
            "tier": "ELEVATED",
            "label": "Substandard Risk",
            "premium_adjustment": "+15%",
            "recommendation": "Approve with surcharge — review in 12 months",
            "color": "orange"
        }
    else:
        return {
            "tier": "HIGH",
            "label": "High Risk",
            "premium_adjustment": "+30%",
            "recommendation": "Refer to underwriting — manual review required",
            "color": "red"
        }


def save_result(request_id: str, input_data: dict, result: dict):
    """Persist scoring result to DynamoDB for audit trail."""
    try:
        table = dynamodb.Table(DYNAMODB_TABLE)
        table.put_item(Item={
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": json.dumps(input_data),
            "risk_score": Decimal(str(round(result["risk_score"], 4))),
            "risk_tier": result["risk_tier"]["tier"],
            "premium_adjustment": result["risk_tier"]["premium_adjustment"],
            "ttl": int(datetime.now(timezone.utc).timestamp()) + (90 * 24 * 60 * 60)  # 90-day TTL
        })
        logger.info(f"Result saved to DynamoDB: {request_id}")
    except Exception as e:
        logger.error(f"DynamoDB write failed: {e}")


def send_high_risk_alert(request_id: str, risk_score: float, input_data: dict):
    """Trigger SNS alert for high-risk assessments."""
    if not SNS_TOPIC_ARN:
        return
    try:
        message = {
            "alert_type": "HIGH_RISK_DETECTED",
            "request_id": request_id,
            "risk_score": risk_score,
            "policy_type": input_data.get("policy_type"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_required": "Manual underwriting review recommended"
        }
        sns_client.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"🚨 High Risk Alert — Score: {risk_score:.2f}",
            Message=json.dumps(message, indent=2)
        )
        logger.info(f"High risk alert sent for request: {request_id}")
    except Exception as e:
        logger.error(f"SNS publish failed: {e}")


def lambda_handler(event: dict, context) -> dict:
    """Main Lambda entry point."""
    import uuid

    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} | Path: {event.get('path')} | Method: {event.get('httpMethod')}")

    # CORS headers
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "X-Request-ID": request_id
    }

    # Handle preflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    # Health check endpoint
    if event.get("path") == "/health":
        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({"status": "healthy", "version": "1.0.0"})
        }

    try:
        # Parse request body
        body = json.loads(event.get("body") or "{}")

        # Validate input
        is_valid, error_msg = validate_input(body)
        if not is_valid:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({
                    "error": "Validation Error",
                    "message": error_msg,
                    "request_id": request_id
                })
            }

        # Load model and score
        model, scaler = load_model()
        features = build_features(body)

        if scaler:
            features = scaler.transform(features)

        risk_score = float(model.predict_proba(features)[0][1])
        risk_tier = get_risk_tier(risk_score)

        result = {
            "request_id": request_id,
            "risk_score": round(risk_score, 4),
            "risk_tier": risk_tier,
            "model_version": "1.0.0",
            "scored_at": datetime.now(timezone.utc).isoformat()
        }

        # Persist to DynamoDB
        save_result(request_id, body, result)

        # Alert if high risk
        if risk_score >= HIGH_RISK_THRESHOLD:
            send_high_risk_alert(request_id, risk_score, body)

        logger.info(f"Request {request_id} scored: {risk_score:.4f} ({risk_tier['tier']})")

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(result)
        }

    except json.JSONDecodeError:
        return {
            "statusCode": 400,
            "headers": headers,
            "body": json.dumps({"error": "Invalid JSON in request body", "request_id": request_id})
        }
    except Exception as e:
        logger.error(f"Unhandled error for request {request_id}: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "Internal server error", "request_id": request_id})
        }
