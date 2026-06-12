#!/usr/bin/env bash
# Secure push helper.
# Usage:
#   export GITHUB_TOKEN=ghp_xxx   # set in your shell, NEVER commit it
#   bash scripts/push.sh "commit message"
#
# The token is read from the environment and used only for this single push.
# It is never written to .git/config, logs, or any tracked file.
set -euo pipefail

MSG="${1:-update}"
REPO="github.com/mohebbibrothers/fath-ai.git"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: export GITHUB_TOKEN first (export GITHUB_TOKEN=ghp_...)" >&2
  exit 1
fi

git add -A
git commit -m "$MSG" || echo "nothing to commit"

git push "https://x-access-token:${GITHUB_TOKEN}@${REPO}" "${BRANCH}:${BRANCH}" \
  2>&1 | sed -E "s/${GITHUB_TOKEN}/***REDACTED***/g; s#x-access-token:[^@]*@#***@#g"

echo "pushed ${BRANCH} OK"
