"""
Model Training Script
Trains insurance risk scoring model and uploads artifact to S3
Run locally before deploying the Lambda function
"""

import pickle
import boto3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score
)
from sklearn.pipeline import Pipeline
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic insurance data for demo purposes.
    Replace this with your real CSV data loader.
    """
    np.random.seed(42)

    age = np.random.normal(42, 15, n_samples).clip(18, 85)
    credit_score = np.random.normal(680, 80, n_samples).clip(300, 850)
    claims_history = np.random.poisson(0.8, n_samples).clip(0, 10)
    policy_type = np.random.choice([0, 1, 2, 3, 4], n_samples)
    years_insured = np.random.exponential(5, n_samples).clip(0, 40)
    annual_mileage = np.random.normal(12000, 4000, n_samples).clip(0, 50000)
    property_value = np.random.lognormal(12, 0.5, n_samples)
    prior_cancellation = np.random.binomial(1, 0.08, n_samples)

    # Risk score formula (actuarial-style)
    risk_logit = (
        -3.0
        + 0.02 * (age - 40)
        - 0.005 * (credit_score - 680)
        + 0.4 * claims_history
        + 0.3 * prior_cancellation
        + 0.1 * (annual_mileage / 10000 - 1.2)
        + np.random.normal(0, 0.5, n_samples)
    )
    risk_prob = 1 / (1 + np.exp(-risk_logit))
    high_risk = (risk_prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "credit_score": credit_score,
        "claims_history": claims_history,
        "policy_type": policy_type,
        "years_insured": years_insured,
        "annual_mileage": annual_mileage,
        "property_value": property_value,
        "prior_cancellation": prior_cancellation,
        "high_risk": high_risk
    })

    logger.info(f"Generated {n_samples} samples | High risk rate: {high_risk.mean():.1%}")
    return df


def load_real_data(filepath: str) -> pd.DataFrame:
    """Load and prepare real CSV data."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows from {filepath}")
    return df


def train_model(df: pd.DataFrame) -> dict:
    """Train ensemble model and return artifacts."""
    feature_cols = [
        "age", "credit_score", "claims_history", "policy_type",
        "years_insured", "annual_mileage", "property_value", "prior_cancellation"
    ]
    target_col = "high_risk"

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Try multiple models — pick best
    candidates = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    for name, clf in candidates.items():
        clf.fit(X_train_scaled, y_train)
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        results[name] = {"model": clf, "auc": auc, "accuracy": acc}
        logger.info(f"{name}: AUC={auc:.4f} | Accuracy={acc:.4f}")

    # Select best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best = results[best_name]
    logger.info(f"\n✅ Best model: {best_name} (AUC={best['auc']:.4f})")

    # Full evaluation
    best_model = best["model"]
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "="*50)
    print(f"MODEL EVALUATION — {best_name}")
    print("="*50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "model": best_model,
        "scaler": scaler,
        "model_name": best_name,
        "metrics": {
            "accuracy": best["accuracy"],
            "auc": best["auc"]
        },
        "feature_names": feature_cols
    }


def save_model_local(model_data: dict, path: str = "model/risk_model.pkl"):
    """Save model artifact locally."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_data, f)
    logger.info(f"Model saved locally: {path}")


def upload_model_to_s3(local_path: str, bucket: str, key: str):
    """Upload model artifact to S3 for Lambda to consume."""
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    logger.info(f"✅ Model uploaded to s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and deploy risk scoring model")
    parser.add_argument("--data", type=str, default=None, help="Path to real CSV data (optional)")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--key", type=str, default="models/risk_model.pkl", help="S3 object key")
    parser.add_argument("--samples", type=int, default=5000, help="Synthetic samples if no CSV")
    args = parser.parse_args()

    # Load or generate data
    if args.data:
        df = load_real_data(args.data)
    else:
        logger.info("No CSV provided — using synthetic data")
        df = generate_synthetic_data(args.samples)

    # Train
    model_data = train_model(df)

    # Save + upload
    local_path = "model/risk_model.pkl"
    save_model_local(model_data, local_path)
    upload_model_to_s3(local_path, args.bucket, args.key)

    print(f"\n🚀 Ready to deploy Lambda!")
    print(f"   Model: {model_data['model_name']}")
    print(f"   AUC:   {model_data['metrics']['auc']:.4f}")
    print(f"   S3:    s3://{args.bucket}/{args.key}")
