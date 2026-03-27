#!/usr/bin/env bash
set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"

echo "Using region: ${AWS_REGION}"

echo "Creating compute environment..."
aws --region "${AWS_REGION}" batch create-compute-environment \
  --cli-input-json file://compute-environment.json

echo "Waiting a bit before creating queue..."
sleep 5

echo "Creating job queue..."
aws --region "${AWS_REGION}" batch create-job-queue \
  --cli-input-json file://job-queue.json

echo "Registering job definition..."
aws --region "${AWS_REGION}" batch register-job-definition \
  --cli-input-json file://job-definition.json

echo "Done."
echo
echo "Next:"
echo "  aws --region ${AWS_REGION} batch submit-job \\"
echo "    --job-name icesee-issm-smoke \\"
echo "    --job-queue icesee-online-queue \\"
echo "    --job-definition icesee-online-issm"