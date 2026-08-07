#!/bin/bash
# Mirror all output to the serial console: SSH to TPU VMs is often blocked by
# firewall policy, and the serial log is then the only way to watch boot progress
# (gcloud compute instances get-serial-port-output).
exec > >(tee /var/log/torchtpu-startup.log > /dev/console) 2>&1
set -ex # Enable command tracing and exit on error

echo "Starting TorchTPU Bootloader..."
echo "-----------------------------------"
echo "Project ID: comglitn"
echo "Zone: europe-west4-a"
echo "Pip packages: torch_tpu transformers"
echo "Package index: us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/"
echo "-----------------------------------"

# Ensure internet connectivity
echo "Checking internet connectivity..."
set +e # Allow ping to fail without exiting immediately
for i in $(seq 1 30); do
  echo "Attempt $i/30: Pinging 8.8.8.8..."
  ping -c 1 8.8.8.8
  if [ $? -eq 0 ]; then
    echo "Internet connected."
    break
  fi
  echo "Ping failed, retrying in 5 seconds..."
  sleep 5
  if [ $i -eq 30 ]; then
    echo "ERROR: Internet connectivity failed after multiple retries. Exiting."
    exit 1
  fi
done
set -e # Re-enable exit on error

# The TorchTPU quickstart requires Python 3.12 (the wheels are built for it).
# The jammy ubuntu-accel image ships 3.10, so fall back to deadsnakes if the
# stock archive doesn't carry 3.12.
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
if ! apt-get install -y python3.12 python3.12-venv; then
  echo "python3.12 not in the stock archive; adding the deadsnakes PPA..."
  apt-get install -y software-properties-common
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -y
  apt-get install -y python3.12 python3.12-venv
fi

# No venv (repo standard): python3.12 is a dedicated interpreter separate from
# the image's system python3 (3.10), so installing straight into its
# site-packages gives the same isolation the upstream quickstart's venv is
# after. --break-system-packages (below) covers builds that mark the
# interpreter externally managed; on this single-purpose, self-deleting VM
# there is nothing to protect.
python3.12 -m ensurepip --upgrade

# The torch_tpu wheels live in a private Artifact Registry; authenticate with the
# VM service account's access token from the metadata server (gcloud is not
# installed on the ubuntu-accel image). A fetch that 403s at pip time means the
# VM's service account lacks Artifact Registry reader access to the registry.
echo "Fetching access token from the metadata server..."
# Tracing (set -x) stays off from here on: it would print the token inside the
# pip index URL. The echo statements below narrate progress instead.
set +x
set +e
ACCESS_TOKEN=""
for i in $(seq 1 12); do
  ACCESS_TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])' 2>/dev/null)
  if [ -n "$ACCESS_TOKEN" ]; then
    echo "Access token fetched."
    break
  fi
  echo "Attempt $i/12: could not fetch an access token. Retrying in 10 seconds..."
  sleep 10
done
set -e
if [ -z "$ACCESS_TOKEN" ]; then
  echo "ERROR: could not fetch an access token from the metadata server."
  exit 1
fi

# Install torch_tpu from the authenticated index. Per the quickstart, the
# matching CPU build of torch (2.11) is pulled automatically — NEVER add torch
# itself to the package list.
echo "Installing PyTorch TPU stack: torch_tpu transformers (index token masked)"
set +e
INSTALL_OK=0
for i in $(seq 1 5); do
  echo "Attempt $i/5: python3.12 -m pip install --pre --break-system-packages --index-url https://oauth2accesstoken:***masked***@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/ torch_tpu transformers"
  python3.12 -m pip install --pre --break-system-packages --index-url "https://oauth2accesstoken:$ACCESS_TOKEN@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/" torch_tpu transformers
  if [ $? -eq 0 ]; then
    INSTALL_OK=1
    echo "PyTorch TPU stack installed."
    break
  fi
  echo "pip install failed, retrying in 20 seconds..."
  sleep 20
done
set -e
if [ "$INSTALL_OK" -eq 0 ]; then
  echo "ERROR: Failed to install the PyTorch TPU stack after multiple retries."
  echo "If pip reported 403/401: grant the VM's service account Artifact Registry"
  echo "reader access to the torch_tpu registry, then reset the VM."
  exit 1
fi

# Write the smoke test to a well-known path so it can be rerun any time
# (the MCP agent's verify_pytorch_tpu tool executes this same file over SSH).
# Rules baked in: plain torch only — never `import torch_tpu` (installation
# registers the "tpu" device) and never PyTorch/XLA; bfloat16 is the native MXU
# dtype; dims are multiples of 128 to tile the MXU cleanly; compile with
# backend="tpu"; .cpu() forces materialization.
cat > /opt/torchtpu_smoke.py << 'PYEOF'
import torch

x = torch.randn(128, 128, dtype=torch.bfloat16, device="tpu")
y = torch.randn(128, 128, dtype=torch.bfloat16, device="tpu")
matmul = torch.compile(torch.matmul, backend="tpu")
out = matmul(x, y).cpu()
print("compiled matmul OK:", tuple(out.shape), out.dtype)
PYEOF

echo "Running TorchTPU smoke test (/opt/torchtpu_smoke.py)..."
set +e
SMOKE_OK=0
for i in $(seq 1 3); do
  python3.12 /opt/torchtpu_smoke.py
  if [ $? -eq 0 ]; then
    SMOKE_OK=1
    break
  fi
  echo "Smoke test failed (attempt $i/3), retrying in 20 seconds..."
  sleep 20
done
set -e
if [ "$SMOKE_OK" -eq 0 ]; then
  echo "ERROR: TorchTPU smoke test failed — torch cannot see or compile for the TPU."
  echo "Inspect /var/log/torchtpu-startup.log on the VM or rerun /opt/torchtpu_smoke.py."
  exit 1
fi

echo "Run torch workloads with python3.12 (the system python3 stays untouched)."
echo "TorchTPU environment ready."
