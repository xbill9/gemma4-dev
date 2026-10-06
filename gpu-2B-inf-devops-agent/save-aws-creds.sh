#!/bin/bash
# This script exports active AWS credentials and saves them to .aws_creds
# for use by the Makefile and other tools.

echo "Exporting AWS credentials..."
CREDS=$(aws configure export-credentials 2>/dev/null)

if [ $? -ne 0 ]; then
    echo "Error: Failed to export AWS credentials."
    echo "Please ensure you are authenticated (e.g., run 'aws sso login' or 'aws configure')."
fi

ACCESS_KEY=$(echo $CREDS | jq -r .AccessKeyId)
SECRET_KEY=$(echo $CREDS | jq -r .SecretAccessKey)
SESSION_TOKEN=$(echo $CREDS | jq -r .SessionToken)

# Create/Overwrite .aws_creds file
{
    echo "AWS_ACCESS_KEY_ID=$ACCESS_KEY"
    echo "AWS_SECRET_ACCESS_KEY=$SECRET_KEY"
    if [ "$SESSION_TOKEN" != "null" ] && [ -n "$SESSION_TOKEN" ]; then
        echo "AWS_SESSION_TOKEN=$SESSION_TOKEN"
    fi
} > .aws_creds

echo "Successfully saved credentials to .aws_creds"
echo "The Makefile will now automatically use these for deployments."
