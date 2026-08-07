#!/usr/bin/env bash
# Compatibility helper: AWS uses its standard credential provider chain, not ADC.
set -euo pipefail
echo "This project targets AWS; Google Application Default Credentials are not used."
echo "Configure AWS SSO/profile credentials, then run: aws sts get-caller-identity"
