#!/usr/bin/env bash
# Deploy (or update) the basin-precip Lambda + its daily EventBridge schedule.
#
# Mirrors the buzzards-bay-foraging Lambda: container image on ECR, reads R2
# creds from the same SSM path (/buzzards-bay-foraging/r2/*), EventBridge
# cron target. All steps are idempotent -- safe to re-run after a code change
# (re-run to push a new image + update-function-code) or after a partial
# failure.
#
#   AWS_PROFILE=osc REGION=us-west-2 ./deploy_lambda.sh
set -euo pipefail
cd "$(dirname "$0")"

REGION="${REGION:-us-west-2}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REPO="basin-precip"
ROLE="basin-precip-lambda"
FUNC="basin-precip"
RULE="basin-precip-daily"
SCHEDULE="cron(30 13 * * ? *)"   # same time as buzzards-bay-foraging-daily
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO"

echo "=== ECR repository ==="
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

echo "=== build + push image ==="
docker build -f Dockerfile.lambda -t "$REPO:latest" .
docker tag "$REPO:latest" "$IMAGE:latest"
docker push "$IMAGE:latest"
DIGEST="$(aws ecr describe-images --region "$REGION" --repository-name "$REPO" \
  --image-ids imageTag=latest --query 'imageDetails[0].imageDigest' --output text)"
IMAGE_URI="$IMAGE@$DIGEST"
echo "image: $IMAGE_URI"

echo "=== IAM role ==="
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]
  }' >/dev/null
fi
aws iam attach-role-policy --role-name "$ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
# Same SSM path (and same underlying R2 keys) as the foraging Lambda's r2-ssm-read policy.
aws iam put-role-policy --role-name "$ROLE" --policy-name r2-ssm-read --policy-document '{
  "Version": "2012-10-17",
  "Statement": [{"Effect": "Allow", "Action": "ssm:GetParameter",
                 "Resource": "arn:aws:ssm:'"$REGION"':'"$ACCOUNT"':parameter/buzzards-bay-foraging/r2/*"}]
}'
ROLE_ARN="arn:aws:iam::$ACCOUNT:role/$ROLE"

echo "=== Lambda function ==="
if aws lambda get-function --region "$REGION" --function-name "$FUNC" >/dev/null 2>&1; then
  aws lambda update-function-code --region "$REGION" --function-name "$FUNC" \
    --image-uri "$IMAGE_URI" >/dev/null
  aws lambda wait function-updated --region "$REGION" --function-name "$FUNC"
else
  # A just-created role isn't always immediately assumable; retry briefly.
  for i in $(seq 1 6); do
    aws lambda create-function --region "$REGION" --function-name "$FUNC" \
      --package-type Image --code ImageUri="$IMAGE_URI" --role "$ROLE_ARN" \
      --timeout 180 --memory-size 1024 --architectures x86_64 >/dev/null 2>&1 && break
    echo "  role not yet assumable, retrying in 10s ..."; sleep 10
  done
  aws lambda wait function-active --region "$REGION" --function-name "$FUNC"
fi

echo "=== EventBridge schedule ==="
aws events put-rule --region "$REGION" --name "$RULE" --schedule-expression "$SCHEDULE" --state ENABLED >/dev/null
FUNC_ARN="arn:aws:lambda:$REGION:$ACCOUNT:function:$FUNC"
aws events put-targets --region "$REGION" --rule "$RULE" --targets "Id=1,Arn=$FUNC_ARN" >/dev/null
aws lambda add-permission --region "$REGION" --function-name "$FUNC" \
  --statement-id "${RULE}-invoke" --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:$REGION:$ACCOUNT:rule/$RULE" >/dev/null 2>&1 || true  # already granted

echo
echo "done: $FUNC scheduled by $RULE at $SCHEDULE (same time as buzzards-bay-foraging-daily)"
