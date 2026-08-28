#!/bin/bash
# This script exports active AWS credentials and saves them to .aws_creds
# for use by the Makefile and other tools.
#
# Usage: save-aws-creds.sh [output-path]     (default: ./.aws_creds)
#        FORCE=1 save-aws-creds.sh           (skip the gitignore guard)

set -euo pipefail

OUT="${1:-.aws_creds}"

# Guard: refuse to write credentials somewhere git would offer to commit them.
# `git check-ignore` exits 0 when the path is ignored, 1 when it is tracked or
# merely untracked, and 128 when the path is outside this repository. Only 1 is
# a problem -- test for it exactly, because treating "not zero" as "not ignored"
# also rejects every path outside the repo, which is the documented way out.
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    ignored=0
    git check-ignore -q "$OUT" 2>/dev/null || ignored=$?
    if [ "$ignored" -eq 1 ] && [ "${FORCE:-}" != "1" ]; then
        echo "Error: '$OUT' is inside a git work tree and is not gitignored."
        echo "Add it to .gitignore, pass a path outside the repo, or set FORCE=1."
        exit 1
    fi
fi

echo "Exporting AWS credentials..."
if ! CREDS=$(aws configure export-credentials); then
    echo "Error: Failed to export AWS credentials."
    echo "Please ensure you are authenticated (e.g., run 'aws sso login' or 'aws configure')."
    exit 1
fi

ACCESS_KEY=$(echo "$CREDS" | jq -r .AccessKeyId)
SECRET_KEY=$(echo "$CREDS" | jq -r .SecretAccessKey)
SESSION_TOKEN=$(echo "$CREDS" | jq -r .SessionToken)

for pair in "AccessKeyId:$ACCESS_KEY" "SecretAccessKey:$SECRET_KEY"; do
    if [ -z "${pair#*:}" ] || [ "${pair#*:}" = "null" ]; then
        echo "Error: AWS returned no ${pair%%:*}."
        echo "The credential output was not in the expected format; refusing to write $OUT."
        exit 1
    fi
done

# Restrict permissions before writing: the redirect below truncates an existing
# file without resetting its mode, so chmod must happen on every run.
: > "$OUT"
chmod 600 "$OUT"

{
    echo "AWS_ACCESS_KEY_ID=$ACCESS_KEY"
    echo "AWS_SECRET_ACCESS_KEY=$SECRET_KEY"
    if [ "$SESSION_TOKEN" != "null" ] && [ -n "$SESSION_TOKEN" ]; then
        echo "AWS_SESSION_TOKEN=$SESSION_TOKEN"
    fi
} > "$OUT"

if [ "$SESSION_TOKEN" = "null" ] || [ -z "$SESSION_TOKEN" ]; then
    echo "Warning: no session token -- these are long-lived user keys, not a role."
fi

echo "Successfully saved credentials to $OUT"
echo "The Makefile will now automatically use these for deployments."
