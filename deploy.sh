#!/usr/bin/env bash
set -euo pipefail

SPACE_NAME="${1:-}"
if [[ -z "$SPACE_NAME" ]]; then
  echo "Usage: bash deploy.sh <username/space-name>"
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "Error: run from a git repository"
  exit 1
fi

HF_REMOTE="hf"
DEPLOY_BRANCH="deploy"
SPACE_URL="https://huggingface.co/spaces/${SPACE_NAME}"

if git remote | grep -q "^${HF_REMOTE}$"; then
  git remote set-url "${HF_REMOTE}" "${SPACE_URL}"
else
  git remote add "${HF_REMOTE}" "${SPACE_URL}"
fi

if ! git show-ref --verify --quiet "refs/heads/${DEPLOY_BRANCH}"; then
  git checkout -b "${DEPLOY_BRANCH}"
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [[ "$CURRENT_BRANCH" != "$DEPLOY_BRANCH" ]]; then
  git checkout "$DEPLOY_BRANCH"
fi

git add .
if ! git diff --staged --quiet; then
  git commit -m "Deploy RTSS diff viewer"
fi

git push "${HF_REMOTE}" "${DEPLOY_BRANCH}:main"

echo "Deployment complete: ${SPACE_URL}"
