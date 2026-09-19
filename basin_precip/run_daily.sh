#!/usr/bin/env bash
# Daily refresh of the per-basin MRMS precip time series on Cloudflare R2.
#
# Launches a us-west-2 VM (OSC AWS account), recomputes the two most recent
# complete report dates -- today's and yesterday's, so a late MRMS revision to
# yesterday is picked up -- merges them into the R2 parquet, and tears the VM
# down. ~3 min wall, well under a cent of EC2.
#
# Schedule it for after 7am ET + MRMS lag (the foraging Lambda fires at 13:30 UTC).
#
# Needs SkyPilot >= 0.13 (skypilot-aws env) and AWS profile `osc`, which also
# holds the R2 keys in SSM (/buzzards-bay-foraging/r2/*).
#   ./run_daily.sh
#   SEL="--dates 2026-09-18" R2_KEY=some/test.parquet ./run_daily.sh   # override
set -euo pipefail
cd "$(dirname "$0")"

export PATH="${SKY_BIN:-$HOME/miniforge3/envs/skypilot-aws/bin}:$PATH"
export AWS_PROFILE="${AWS_PROFILE:-osc}"
CLUSTER="${CLUSTER:-bb-precip-daily}"
SEL="${SEL:---last 2}"
R2_KEY="${R2_KEY:-buzzards-bay-precip/basins_v2_precip_ts.parquet}"

ssm() { aws ssm get-parameter --region us-west-2 --name "/buzzards-bay-foraging/r2/$1" \
          --with-decryption --query Parameter.Value --output text; }
export R2_ACCESS_KEY_ID="$(ssm access-key-id)"
export R2_SECRET_ACCESS_KEY="$(ssm secret-access-key)"

# Never leave the VM running, whatever happens below.
trap 'sky down "$CLUSTER" -y >/dev/null 2>&1 || true' EXIT

echo "$(date -u +%FT%TZ) launching $CLUSTER: $SEL -> $R2_KEY"
sky launch -c "$CLUSTER" basin-precip.sky.yaml -i 10 --down -y \
  --env "SEL=$SEL" --env "R2_KEY=$R2_KEY" --env "OUT=out/ts.parquet" \
  --env R2_ACCESS_KEY_ID --env R2_SECRET_ACCESS_KEY

sky down "$CLUSTER" -y >/dev/null
# Termination is asynchronous: wait (up to 2 min) for nothing to be pending/running.
for _ in $(seq 24); do
  left=$(aws ec2 describe-instances --region us-west-2 \
    --filters "Name=instance-state-name,Values=pending,running" \
              "Name=tag:Name,Values=sky-${CLUSTER}-*" \
    --query 'length(Reservations[].Instances[])' --output text)
  [ "$left" = 0 ] && break
  sleep 5
done
echo "$(date -u +%FT%TZ) done; VMs still running: $left"
[ "$left" = 0 ]
