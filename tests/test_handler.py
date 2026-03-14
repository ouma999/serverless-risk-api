"""
Test Suite — Serverless Insurance Risk Scoring API
Tests Lambda handler logic without AWS dependencies using mocks
Run: pytest tests/ -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../lambda"))


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def valid_payload():
    return {
        "age": 45,
        "credit_score": 680,
        "claims_history": 1,
        "policy_type": "auto",
        "years_insured": 5,
        "annual_mileage": 12000,
        "prior_cancellation": False
    }


@pytest.fixture
def high_risk_payload():
    return {
        "age": 23,
        "credit_score": 420,
        "claims_history": 5,
        "policy_type": "auto",
        "years_insured": 1,
        "annual_mileage": 25000,
        "prior_cancellation": True
    }


@pytest.fixture
def api_gateway_event(valid_payload):
    return {
        "httpMethod": "POST",
        "path": "/score-risk",
        "body": json.dumps(valid_payload),
        "headers": {"Content-Type": "application/json"}
    }


def make_mock_model(score=0.45):
    """Create a mock sklearn model that returns a fixed probability."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[1 - score, score]])
    model.predict.return_value = np.array([1 if score > 0.5 else 0])
    return model


# ─────────────────────────────────────────────
# Unit Tests — Input Validation
# ─────────────────────────────────────────────

class TestValidation:

    def test_valid_payload_passes(self, valid_payload):
        from handler import validate_input
        is_valid, msg = validate_input(valid_payload)
        assert is_valid is True
        assert msg == ""

    def test_missing_age_fails(self, valid_payload):
        from handler import validate_input
        del valid_payload["age"]
        is_valid, msg = validate_input(valid_payload)
        assert is_valid is False
        assert "age" in msg

    def test_missing_credit_score_fails(self, valid_payload):
        from handler import validate_input
        del valid_payload["credit_score"]
        is_valid, msg = validate_input(valid_payload)
        assert is_valid is False

    def test_age_out_of_range_fails(self, valid_payload):
        from handler import validate_input
        valid_payload["age"] = 150
        is_valid, msg = validate_input(valid_payload)
        assert is_valid is False
        assert "age" in msg

    def test_credit_score_below_minimum_fails(self, valid_payload):
        from handler import validate_input
        valid_payload["credit_score"] = 200
        is_valid, msg = validate_input(valid_payload)
        assert is_valid is False

    def test_invalid_policy_type_fails(self, valid_payload):
        from handler import validate_input
        valid_payload["policy_type"] = "spaceship"
        is_valid, msg = validate_input(valid_payload)
        assert is_valid is False
        assert "policy_type" in msg

    def test_all_valid_policy_types_pass(self, valid_payload):
        from handler import validate_input
        for pt in ["auto", "home", "life", "health", "commercial"]:
            valid_payload["policy_type"] = pt
            is_valid, _ = validate_input(valid_payload)
            assert is_valid is True


# ─────────────────────────────────────────────
# Unit Tests — Risk Tier Assignment
# ─────────────────────────────────────────────

class TestRiskTier:

    def test_low_risk_tier(self):
        from handler import get_risk_tier
        result = get_risk_tier(0.10)
        assert result["tier"] == "LOW"
        assert "-" in result["premium_adjustment"]

    def test_moderate_risk_tier(self):
        from handler import get_risk_tier
        result = get_risk_tier(0.35)
        assert result["tier"] == "MODERATE"
        assert result["premium_adjustment"] == "0%"

    def test_elevated_risk_tier(self):
        from handler import get_risk_tier
        result = get_risk_tier(0.60)
        assert result["tier"] == "ELEVATED"
        assert "+" in result["premium_adjustment"]

    def test_high_risk_tier(self):
        from handler import get_risk_tier
        result = get_risk_tier(0.85)
        assert result["tier"] == "HIGH"

    def test_boundary_at_025(self):
        from handler import get_risk_tier
        assert get_risk_tier(0.249)["tier"] == "LOW"
        assert get_risk_tier(0.250)["tier"] == "MODERATE"

    def test_boundary_at_075(self):
        from handler import get_risk_tier
        assert get_risk_tier(0.749)["tier"] == "ELEVATED"
        assert get_risk_tier(0.750)["tier"] == "HIGH"


# ─────────────────────────────────────────────
# Unit Tests — Feature Engineering
# ─────────────────────────────────────────────

class TestFeatureEngineering:

    def test_feature_shape(self, valid_payload):
        from handler import build_features
        features = build_features(valid_payload)
        assert features.shape == (1, 8)

    def test_prior_cancellation_encoding(self, valid_payload):
        from handler import build_features
        valid_payload["prior_cancellation"] = True
        features = build_features(valid_payload)
        assert features[0, 7] == 1

        valid_payload["prior_cancellation"] = False
        features = build_features(valid_payload)
        assert features[0, 7] == 0

    def test_policy_type_encoding(self, valid_payload):
        from handler import build_features
        valid_payload["policy_type"] = "home"
        features = build_features(valid_payload)
        assert features[0, 3] == 1  # home = 1


# ─────────────────────────────────────────────
# Integration Tests — Lambda Handler
# ─────────────────────────────────────────────

class TestLambdaHandler:

    @patch("handler.save_result")
    @patch("handler.send_high_risk_alert")
    @patch("handler.load_model")
    def test_successful_scoring(self, mock_load, mock_alert, mock_save, api_gateway_event):
        from handler import lambda_handler
        mock_load.return_value = (make_mock_model(0.45), None)

        response = lambda_handler(api_gateway_event, None)
        body = json.loads(response["body"])

        assert response["statusCode"] == 200
        assert "risk_score" in body
        assert "risk_tier" in body
        assert "request_id" in body
        assert 0 <= body["risk_score"] <= 1

    @patch("handler.save_result")
    @patch("handler.send_high_risk_alert")
    @patch("handler.load_model")
    def test_high_risk_triggers_alert(self, mock_load, mock_alert, mock_save, high_risk_payload):
        from handler import lambda_handler
        mock_load.return_value = (make_mock_model(0.90), None)

        event = {
            "httpMethod": "POST",
            "path": "/score-risk",
            "body": json.dumps(high_risk_payload)
        }

        response = lambda_handler(event, None)
        assert response["statusCode"] == 200
        mock_alert.assert_called_once()

    def test_invalid_json_returns_400(self):
        from handler import lambda_handler
        event = {
            "httpMethod": "POST",
            "path": "/score-risk",
            "body": "not valid json{"
        }
        response = lambda_handler(event, None)
        assert response["statusCode"] == 400

    def test_missing_fields_returns_400(self):
        from handler import lambda_handler
        event = {
            "httpMethod": "POST",
            "path": "/score-risk",
            "body": json.dumps({"age": 45})
        }
        response = lambda_handler(event, None)
        assert response["statusCode"] == 400

    def test_health_check_returns_200(self):
        from handler import lambda_handler
        event = {"httpMethod": "GET", "path": "/health"}
        response = lambda_handler(event, None)
        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "healthy"

    def test_options_preflight_returns_200(self):
        from handler import lambda_handler
        event = {"httpMethod": "OPTIONS", "path": "/score-risk"}
        response = lambda_handler(event, None)
        assert response["statusCode"] == 200

    @patch("handler.save_result")
    @patch("handler.send_high_risk_alert")
    @patch("handler.load_model")
    def test_response_has_cors_headers(self, mock_load, mock_alert, mock_save, api_gateway_event):
        from handler import lambda_handler
        mock_load.return_value = (make_mock_model(0.3), None)

        response = lambda_handler(api_gateway_event, None)
        assert "Access-Control-Allow-Origin" in response["headers"]

    @patch("handler.load_model", side_effect=Exception("S3 connection failed"))
    def test_model_load_failure_returns_500(self, mock_load, api_gateway_event):
        from handler import lambda_handler
        response = lambda_handler(api_gateway_event, None)
        assert response["statusCode"] == 500
