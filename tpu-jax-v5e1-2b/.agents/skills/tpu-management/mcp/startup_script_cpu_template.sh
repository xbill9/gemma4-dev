#!/bin/bash
# CPU debug box: the same JAX stack as the TPU VMs, minus libtpu.
#
# Purpose is correctness work, not benchmarking. A host with enough RAM can load
# the real checkpoints (31B at W4A16 is ~16 GB of weights) and run the engine
# through JAX's CPU backend unchanged — `w4a16_impl` defaults to the reference
# path and Pallas auto-switches to interpret mode off-TPU. Architecture bugs are
# reproducible here for cents an hour instead of TPU rates.
#
# Mirror all output to the serial console: SSH is often blocked by firewall policy
# and the serial log is then the only way to watch boot progress.
exec > >(tee /var/log/cpu-debug-startup.log > /dev/console) 2>&1

# set -e is load-bearing, and the ERR trap is installed BEFORE tracing: with -x on,
# the shell echoes the trap definition, and that trace line contains the FAILED
# marker verbatim — which reads as a failure to any log scanner.
set -eu
trap 'rc=$?; echo "JAX-BOOTLOADER: ERROR on line $LINENO (exit $rc)"; echo "JAX-BOOTLOADER: FAILED"; exit $rc' ERR
set -x

echo "Starting CPU debug Bootloader..."
echo "-----------------------------------"
echo "Project ID: {project_id}"
echo "Zone: {zone}"
echo "Python: {python_version}"
echo "Packages: {pip_spec}"
echo "-----------------------------------"

PY=python{python_version}
export DEBIAN_FRONTEND=noninteractive

for i in $(seq 1 30); do
  apt-get update -y && break
  echo "apt-get update retry $i"
  sleep 10
done

# add-apt-repository is not on the base image.
apt-get install -y software-properties-common curl git

# Stock Ubuntu 22.04 ships Python 3.10, which pins JAX to an old release.
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

# No pip ships with a deadsnakes interpreter and the system pip predates
# --break-system-packages, so bootstrap pip into $PY directly. No venv, per this
# repo's standard: the dedicated interpreter provides the isolation.
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
$PY /tmp/get-pip.py
$PY -m pip install --upgrade pip setuptools wheel

# CPU JAX: no libtpu, no -f jax-releases index.
$PY -m pip install --upgrade {pip_spec}

# Assert the stack actually works. Importing jax succeeds in almost any broken
# state, so assert on a real computation and on the CPU device being present.
$PY - <<'PYEOF'
import sys
import jax, jax.numpy as jnp
devs = jax.devices()
print("JAX", jax.__version__, "devices:", devs)
if not any(d.platform == "cpu" for d in devs):
    print("ERROR: no CPU device visible to JAX; got", devs)
    sys.exit(1)
x = jnp.ones((256, 256))
if float((x @ x).sum()) != 256.0 ** 3:
    print("ERROR: CPU matmul produced the wrong result")
    sys.exit(1)
import safetensors, huggingface_hub
print("safetensors", safetensors.__version__, "| hf_hub", huggingface_hub.__version__)
PYEOF

MEM=$(free -g | grep '^Mem:' | tr -s ' ' | cut -d' ' -f2)
echo "host memory: $MEM GiB"
# Measured, not estimated. `load()` holds every tensor of every shard — vision and
# audio towers included — before conversion, and the parameter tree then aliases
# those arrays, so peak RSS tracks the whole checkpoint rather than the text
# weights alone. Add a few GB of XLA:CPU temporaries on top for a forward pass.
echo "  E2B  load + generation : 26 GiB measured peak (6.6 GiB of weights)"
echo "  31B  load only         : 48 GiB measured peak (19 GiB of weights)"
echo "  31B  load + forward    : >64 GiB — OOM-killed on a 64 GiB host"
echo "  26B MoE                : not measured (gated repo)"
echo "  -> 128 GiB to run a 31B, 64 GiB to only load one, 32 GiB for E2B"

$PY -m pip list 2>/dev/null | grep -iE "^(jax|jaxlib|numpy|scipy|ml.dtypes|safetensors|transformers|huggingface) " || true

echo "JAX-BOOTLOADER: CPU environment ready."
