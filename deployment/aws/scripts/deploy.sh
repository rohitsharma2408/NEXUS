#!/usr/bin/env bash
# Builds the API + dashboard images, pushes them to ECR, and forces a new ECS
# deployment so the running services pick up the new image. Mirrors what
# deployment/github_actions/ci-cd.yml does automatically on push to main.
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
source .env 2>/dev/null || true

: "${AWS_REGION:?}"
: "${AWS_ACCOUNT_ID:?}"
: "${ECR_REPO_API:=nexus-api}"
: "${ECR_REPO_DASHBOARD:=nexus-dashboard}"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "Logging into ECR ($ECR_REGISTRY) ..."
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"

echo "Building images ..."
docker build -f deployment/Dockerfile.api -t "${ECR_REGISTRY}/${ECR_REPO_API}:latest" .
docker build -f deployment/Dockerfile.dashboard -t "${ECR_REGISTRY}/${ECR_REPO_DASHBOARD}:latest" .

echo "Pushing images ..."
docker push "${ECR_REGISTRY}/${ECR_REPO_API}:latest"
docker push "${ECR_REGISTRY}/${ECR_REPO_DASHBOARD}:latest"

echo "Forcing new ECS deployments ..."
aws ecs update-service --cluster nexus-cluster --service nexus-api --force-new-deployment --region "$AWS_REGION" >/dev/null
aws ecs update-service --cluster nexus-cluster --service nexus-dashboard --force-new-deployment --region "$AWS_REGION" >/dev/null

echo "Deployed. Check the ALB DNS name from 'terraform output alb_dns_name'."
