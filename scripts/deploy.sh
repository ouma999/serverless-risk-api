#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — One-command deployment for Serverless Risk Scoring API
# Usage: ./scripts/deploy.sh [dev|staging|prod] [aws-profile]
# ─────────────────────────────────────────────────────────────────────────────

set -e  # Exit on any error

# ── Config ────────────────────────────────────────────────────────────────────
ENVIRONMENT=${1:-dev}
AWS_PROFILE=${2:-default}
PROJECT_NAME="risk-scoring-api"
REGION="us-east-1"
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
LAMBDA_ZIP="lambda_package.zip"
ALERT_EMAIL="your@email.com"   # ← Change this

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deploying: ${STACK_NAME}"
echo "  Region:    ${REGION}"
echo "  Profile:   ${AWS_PROFILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Validate CloudFormation template ──────────────────────────────────
echo ""
echo "▶ Step 1/6: Validating CloudFormation template..."
aws cloudformation validate-template \
  --template-body file://cloudformation/template.yaml \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}" > /dev/null

echo "  ✅ Template valid"

# ── Step 2: Deploy CloudFormation Stack ───────────────────────────────────────
echo ""
echo "▶ Step 2/6: Deploying CloudFormation stack..."
aws cloudformation deploy \
  --template-file cloudformation/template.yaml \
  --stack-name "${STACK_NAME}" \
  --parameter-overrides \
      ProjectName="${PROJECT_NAME}" \
      Environment="${ENVIRONMENT}" \
      AlertEmail="${ALERT_EMAIL}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}"

echo "  ✅ Stack deployed"

# ── Step 3: Get outputs from stack ────────────────────────────────────────────
echo ""
echo "▶ Step 3/6: Fetching stack outputs..."
MODEL_BUCKET=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='ModelBucketName'].OutputValue" \
  --output text \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}")

LAMBDA_FUNCTION=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='LambdaFunctionName'].OutputValue" \
  --output text \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}")

API_ENDPOINT=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --query "Stacks[0].Outputs[?OutputKey=='ScoreRiskEndpoint'].OutputValue" \
  --output text \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}")

echo "  Bucket:   ${MODEL_BUCKET}"
echo "  Lambda:   ${LAMBDA_FUNCTION}"
echo "  API:      ${API_ENDPOINT}"

# ── Step 4: Train and upload model ────────────────────────────────────────────
echo ""
echo "▶ Step 4/6: Training model and uploading to S3..."
python model/train.py \
  --bucket "${MODEL_BUCKET}" \
  --key "models/risk_model.pkl"

echo "  ✅ Model uploaded"

# ── Step 5: Package and deploy Lambda code ────────────────────────────────────
echo ""
echo "▶ Step 5/6: Packaging Lambda function..."
cd lambda
pip install -r requirements.txt -t package/ --quiet
cp handler.py package/
cd package && zip -r "../../${LAMBDA_ZIP}" . -x "*.pyc" -x "__pycache__/*" > /dev/null
cd ../..

echo "  Deploying Lambda code..."
aws lambda update-function-code \
  --function-name "${LAMBDA_FUNCTION}" \
  --zip-file "fileb://${LAMBDA_ZIP}" \
  --profile "${AWS_PROFILE}" \
  --region "${REGION}" > /dev/null

rm -f "${LAMBDA_ZIP}"
rm -rf lambda/package/
echo "  ✅ Lambda deployed"

# ── Step 6: Smoke test ────────────────────────────────────────────────────────
echo ""
echo "▶ Step 6/6: Running smoke test..."
sleep 3  # Brief wait for Lambda to update

HEALTH_URL="${API_ENDPOINT/score-risk/health}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}")

if [ "${HTTP_STATUS}" = "200" ]; then
  echo "  ✅ Health check passed (HTTP ${HTTP_STATUS})"
else
  echo "  ⚠️  Health check returned HTTP ${HTTP_STATUS}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  API Endpoint:"
echo "  ${API_ENDPOINT}"
echo ""
echo "  Test it now:"
echo "  curl -X POST ${API_ENDPOINT} \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"age\":45,\"credit_score\":620,\"claims_history\":3,\"policy_type\":\"auto\"}'"
echo ""
