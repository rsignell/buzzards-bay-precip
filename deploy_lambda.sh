#!/usr/bin/env bash
# Push a code change to the buzzards-bay-foraging Lambda.
#
# The Lambda runs its own copies of mrms_moisture.py, score_species.py and
# make_foraging_app.py baked into a container image (Dockerfile.lambda), so a
# git push does NOT change what runs daily -- re-run this after editing any of
# them, or the static inputs (s2/, soil/, land/).
#
# The function, its IAM role (buzzards-bay-foraging-lambda), the ECR repo and
# the buzzards-bay-foraging-daily EventBridge rule already exist; this only
# builds, pushes and repoints the function. Pass --invoke to run it once
# afterwards and republish the app now instead of at the next 13:30 UTC run.
#
#   ./deploy_lambda.sh [--invoke]        (AWS profile defaults to osc)
set -euo pipefail
cd "$(dirname "$0")"

export AWS_PROFILE="${AWS_PROFILE:-osc}"
REGION="${REGION:-us-west-2}"
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REPO="buzzards-bay-foraging"
FUNC="buzzards-bay-foraging"
IMAGE="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO"

for f in s2/oak_index.tif soil/soil_awc25.tif soil/soil_awc.tif land/public_land.tif; do
  [[ -f $f ]] || { echo "missing $f -- the image bakes it in" >&2; exit 1; }
done

echo "=== build + push image ==="
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
docker build -f Dockerfile.lambda -t "$REPO:latest" .
docker tag "$REPO:latest" "$IMAGE:latest"
docker push "$IMAGE:latest"
DIGEST="$(aws ecr describe-images --region "$REGION" --repository-name "$REPO" \
  --image-ids imageTag=latest --query 'imageDetails[0].imageDigest' --output text)"
IMAGE_URI="$IMAGE@$DIGEST"
echo "image: $IMAGE_URI"

echo "=== update function ==="
aws lambda update-function-code --region "$REGION" --function-name "$FUNC" \
  --image-uri "$IMAGE_URI" >/dev/null
aws lambda wait function-updated --region "$REGION" --function-name "$FUNC"
echo "$FUNC now runs $IMAGE_URI"

if [[ "${1:-}" == "--invoke" ]]; then
  echo "=== invoke ==="
  out="$(mktemp)"
  aws lambda invoke --region "$REGION" --function-name "$FUNC" \
    --cli-read-timeout 900 "$out" >/dev/null
  cat "$out"; echo
fi
