#!/bin/bash
# Bare JAX-on-TPU dev VM: no docker, no vLLM, no Hugging Face token.
#
# Mirror all output to the serial console: SSH to TPU VMs is often blocked by
# firewall policy, and the serial log is then the only way to watch boot progress
# (gcloud compute instances get-serial-port-output).
exec > >(tee /var/log/jax-startup.log > /dev/console) 2>&1

# set -e is load-bearing. An earlier hand-rolled version of this script omitted
# it, its pip step failed, and it still printed a success marker — the VM looked
# ready and had no JAX on it. Never emit the ready marker off the happy path.
set -eu
# Install the ERR trap BEFORE enabling -x. With tracing on, the trap definition
# itself is echoed to the log, and that trace line contains the literal FAILED
# marker — so a log scanner would report failure on a perfectly healthy boot.
trap 'rc=$?; echo "JAX-BOOTLOADER: ERROR on line $LINENO (exit $rc)"; echo "JAX-BOOTLOADER: FAILED"; exit $rc' ERR
set -x

echo "Starting JAX TPU Bootloader..."
echo "-----------------------------------"
echo "Project ID: {project_id}"
echo "Zone: {zone}"
echo "Python: {python_version}"
echo "JAX spec: {jax_pip_spec}"
echo "-----------------------------------"

PY=python{python_version}
export DEBIAN_FRONTEND=noninteractive

# apt on a fresh VM races cloud-init's own apt runs; retry rather than die.
for i in $(seq 1 30); do
  apt-get update -y && break
  echo "apt-get update retry $i"
  sleep 10
done

# software-properties-common provides add-apt-repository, which is NOT installed
# on the accelerator image by default.
# Install the newest compiler/build tools published by the configured Ubuntu
# repositories. Native extensions then build against a current toolchain.
apt-get install -y software-properties-common curl build-essential clang cmake ninja-build

# The stock Ubuntu 22.04 interpreter is 3.10 and pins JAX to an old release.
# deadsnakes carries current CPython for jammy.
if ! command -v $PY >/dev/null 2>&1; then
  add-apt-repository -y ppa:deadsnakes/ppa
  for i in $(seq 1 30); do
    apt-get update -y && break
    echo "apt-get update retry $i"
    sleep 10
  done
  apt-get install -y $PY $PY-dev
fi
$PY --version

# Ubuntu ships no pip for a deadsnakes interpreter, and the system pip (22.x) is
# too old for --break-system-packages, so bootstrap pip into $PY directly.
# No venv, per this repo's standard: the dedicated interpreter is the isolation.
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
$PY /tmp/get-pip.py
$PY -m pip install --upgrade pip setuptools wheel packaging

# libtpu comes from the JAX releases index, not PyPI.
$PY -m pip install --upgrade {jax_pip_spec} \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# JAX's TPU extra pins the libtpu version it was released against. This project
# deliberately tracks the newest published TPU runtime instead: install it after
# JAX and bypass that conservative transitive pin. Device verification below is
# the compatibility gate.
$PY -m pip install --upgrade --no-deps libtpu \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
if [ -n "{jax_pip_extras}" ]; then
  $PY -m pip install --upgrade --upgrade-strategy eager {jax_pip_extras} || echo "WARNING: extras install failed (non-fatal)"
fi

# Prove the accelerator is actually visible. Importing jax succeeds on a host
# with no TPU backend, so assert on the device list, not on the import.
$PY - <<'PYEOF'
import sys
import jax
devs = jax.devices()
print("JAX", jax.__version__, "devices:", devs)
if not any(d.platform == "tpu" for d in devs):
    print("ERROR: no TPU device visible to JAX; got", devs)
    sys.exit(1)
print("TPU chips visible:", len(devs))
PYEOF

$PY -m pip list 2>/dev/null | grep -iE "^(jax|jaxlib|libtpu|numpy|scipy|ml.dtypes|safetensors|huggingface.hub|transformers|tokenizers|sentencepiece|jinja2) " || true

# libtpu creates /tmp/tpu_logs as root here; without this every later non-root
# run spams "Could not open the log file ... Permission denied".
chmod -R 1777 /tmp/tpu_logs 2>/dev/null || true

echo "JAX-BOOTLOADER: TPU environment ready."
