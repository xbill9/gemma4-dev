#!/bin/bash
# Mirror all output to the serial console: SSH to TPU VMs is often blocked by
# firewall policy, and the serial log is then the only way to watch boot progress
# (gcloud compute instances get-serial-port-output).
exec > >(tee /var/log/tpu-startup.log > /dev/console) 2>&1
set -ex # Enable command tracing and exit on error

echo "Starting the PyTorch TPU backend Bootloader..."
echo "-----------------------------------"
echo "Project ID: {project_id}"
echo "Zone: {zone}"
echo "Pip packages: {torch_pip_spec}"
echo "Package index: {torch_index}"
echo "Wheel mirror: {wheels_gcs}"
echo "Pip extras: {torch_pip_extras}"
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

# The the PyTorch TPU backend quickstart requires Python 3.12 (the wheels are built for it).
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

# The backend wheels live in a private Artifact Registry; authenticate with the
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

# Preferred path: the project-owned GCS wheel mirror (per the PyTorch TPU backend GDE
# quickstart's wheel install: custom torch base first, then the backend
# extension; remaining deps resolve from public PyPI). The VM's service
# account can be granted read on our own bucket — unlike Google's Artifact
# Registry, where the default compute SA typically 403s.
INSTALL_OK=0
WHEELS_GCS="{wheels_gcs}"
if [ -n "$WHEELS_GCS" ]; then
  set +e
  GCS_BUCKET=$(echo "$WHEELS_GCS" | sed 's|^gs://||; s|/.*$||')
  echo "Trying wheel mirror gs://$GCS_BUCKET ..."
  WHEEL_NAMES=$(curl -sf -H "Authorization: Bearer $ACCESS_TOKEN" \
    "https://storage.googleapis.com/storage/v1/b/$GCS_BUCKET/o?alt=json" \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(sorted(i["name"] for i in d.get("items", []) if i["name"].endswith(".whl"))))' 2>/dev/null)
  if [ -n "$WHEEL_NAMES" ]; then
    mkdir -p /tmp/tpu-wheels
    DL_OK=1
    for NAME in $WHEEL_NAMES; do
      ENC=$(python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$NAME")
      echo "Downloading $NAME from the mirror..."
      curl -sf -H "Authorization: Bearer $ACCESS_TOKEN" -o "/tmp/tpu-wheels/$NAME" \
        "https://storage.googleapis.com/storage/v1/b/$GCS_BUCKET/o/$ENC?alt=media" || DL_OK=0
    done
    if [ "$DL_OK" -eq 1 ]; then
      # The base torch wheel must install before the backend extension wheel.
      BASE_WHL=$(ls /tmp/tpu-wheels/torch-*.whl 2>/dev/null | head -1)
      EXT_WHL=$(ls /tmp/tpu-wheels/*.whl 2>/dev/null | grep -v "/torch-" | head -1)
      python3.12 -m pip install --break-system-packages "$BASE_WHL" \
        && python3.12 -m pip install --break-system-packages "$EXT_WHL" \
        && INSTALL_OK=1
    fi
  fi
  if [ "$INSTALL_OK" -eq 1 ]; then
    echo "PyTorch TPU stack installed from the GCS wheel mirror."
  else
    echo "Wheel mirror unavailable or failed; falling back to the pip index."
  fi
  set -e
fi

# Fallback: the authenticated Artifact Registry index. Per the quickstart, the
# matching CPU build of torch (2.11) is pulled automatically — NEVER add torch
# itself to the package list.
if [ "$INSTALL_OK" -eq 0 ]; then
  echo "Installing PyTorch TPU stack: {torch_pip_spec} (index token masked)"
  set +e
  for i in $(seq 1 5); do
    echo "Attempt $i/5: python3.12 -m pip install --pre --break-system-packages --index-url https://oauth2accesstoken:***masked***@{torch_index} {torch_pip_spec}"
    python3.12 -m pip install --pre --break-system-packages --index-url "https://oauth2accesstoken:$ACCESS_TOKEN@{torch_index}" {torch_pip_spec}
    if [ $? -eq 0 ]; then
      INSTALL_OK=1
      echo "PyTorch TPU stack installed."
      break
    fi
    echo "pip install failed, retrying in 20 seconds..."
    sleep 20
  done
  set -e
fi
if [ "$INSTALL_OK" -eq 0 ]; then
  echo "ERROR: Failed to install the PyTorch TPU stack after multiple retries."
  echo "Fixes: upload the two wheels to $WHEELS_GCS (grant the VM SA"
  echo "roles/storage.objectViewer), or grant it Artifact Registry reader on"
  echo "the backend registry, then reset the VM."
  exit 1
fi

# QAT / kernel-dev extras from public PyPI (transformers, compressed-tensors,
# jinja2>=3.1, jax for Pallas). Best-effort: the TPU stack itself is already
# verified below, and workloads can re-run this install if needed.
if [ -n "{torch_pip_extras}" ]; then
  echo "Installing extras: {torch_pip_extras}"
  set +e
  python3.12 -m pip install --break-system-packages {torch_pip_extras}
  if [ $? -ne 0 ]; then
    echo "WARNING: extras install failed; continuing without them."
  fi
  set -e
fi

# Write the smoke test to a well-known path so it can be rerun any time
# (the MCP agent's verify_pytorch_tpu tool executes this same file over SSH).
# Rules baked in: plain torch only — never import the backend package directly
# (installing it registers the "tpu" device) and never PyTorch/XLA; bfloat16 is the native MXU
# dtype; dims are multiples of 128 to tile the MXU cleanly; compile with
# backend="tpu"; .cpu() forces materialization.
cat > /opt/tpu_smoke.py << 'PYEOF'
import torch

x = torch.randn(128, 128, dtype=torch.bfloat16, device="tpu")
y = torch.randn(128, 128, dtype=torch.bfloat16, device="tpu")
matmul = torch.compile(torch.matmul, backend="tpu")
out = matmul(x, y).cpu()
print("compiled matmul OK:", tuple(out.shape), out.dtype)
PYEOF

echo "Running the PyTorch TPU backend smoke test (/opt/tpu_smoke.py)..."
set +e
SMOKE_OK=0
for i in $(seq 1 3); do
  python3.12 /opt/tpu_smoke.py
  if [ $? -eq 0 ]; then
    SMOKE_OK=1
    break
  fi
  echo "Smoke test failed (attempt $i/3), retrying in 20 seconds..."
  sleep 20
done
set -e
if [ "$SMOKE_OK" -eq 0 ]; then
  echo "ERROR: the PyTorch TPU backend smoke test failed — torch cannot see or compile for the TPU."
  echo "Inspect /var/log/tpu-startup.log on the VM or rerun /opt/tpu_smoke.py."
  exit 1
fi

# Operator-supplied environment exports, written to /etc/environment so that
# non-interactive SSH commands (how the MCP tools and benchmarks run) inherit
# them. Backend env vars lock at init, so they must land before any torch
# import. This is deliberately a generic KEY=VALUE passthrough: the backend's
# own variable names are fixed by its runtime and are not committed here, so
# the operator supplies them (newline- or space-separated) via
# TPU_BACKEND_ENV_EXPORTS. The persistent compile cache — tier-2 in /dev/shm
# plus a tier-3 bucket root, worth 19-70s of warmup per repeat run — is
# configured this way.
ENV_EXPORTS="{env_exports}"
if [ -n "$ENV_EXPORTS" ]; then
  echo "Writing operator-supplied environment exports to /etc/environment..."
  for KV in $ENV_EXPORTS; do
    case "$KV" in
      *=*) echo "$KV" >> /etc/environment; echo "  set ${{KV%%=*}}" ;;
      *) echo "  skipping malformed export (no '='): $KV" ;;
    esac
  done
fi

echo "Run torch workloads with python3.12 (the system python3 stays untouched)."
echo "TPU environment ready."
