#!/bin/bash
# Bare Rust-on-TPU dev VM: a Rust toolchain, a build environment, and libtpu.
# No docker, no vLLM, no Hugging Face token, and no Python in the serving path —
# CPython appears here only as the delivery mechanism for the libtpu wheel, which
# is not published anywhere else.
#
# This prepares the ENVIRONMENT and stops. It does not fetch the engine and does
# not serve: the source is uploaded and built by `deploy_jaxrust_engine`, and the
# thing that proves the chip works is `verify_rust_tpu` running the probe binary.
# Splitting it that way keeps a compile error and a capacity failure from arriving
# through the same channel, which on the JAX path they did.
#
# Mirror all output to the serial console: SSH to TPU VMs is often blocked by
# firewall policy, and the serial log is then the only way to watch boot progress
# (gcloud compute instances get-serial-port-output).
exec > >(tee /var/log/jaxrust-startup.log > /dev/console) 2>&1

# set -e is load-bearing. An earlier hand-rolled version of the JAX script omitted
# it, its pip step failed, and it still printed a success marker — the VM looked
# ready and had nothing on it. Never emit the ready marker off the happy path.
set -eu
# Install the ERR trap BEFORE enabling -x. With tracing on, the trap definition
# itself is echoed to the log, and that trace line contains the literal FAILED
# marker — so a log scanner would report failure on a perfectly healthy boot.
trap 'rc=$?; echo "JAXRUST-BOOTLOADER: ERROR on line $LINENO (exit $rc)"; echo "JAXRUST-BOOTLOADER: FAILED"; exit $rc' ERR
set -x

echo "Starting Rust/XLA TPU Bootloader..."
echo "-----------------------------------"
echo "Project ID: {project_id}"
echo "Zone: {zone}"
echo "Rust toolchain: {rust_toolchain}"
echo "libtpu spec: {libtpu_spec}"
echo "-----------------------------------"

export DEBIAN_FRONTEND=noninteractive
export CARGO_HOME=/opt/rust/cargo
export RUSTUP_HOME=/opt/rust/rustup

# apt on a fresh VM races cloud-init's own apt runs; retry rather than die.
for i in $(seq 1 30); do
  apt-get update -y && break
  echo "apt-get update retry $i"
  sleep 10
done

# protobuf-compiler is NOT optional and its absence is not obvious: pjrt-sys runs
# prost-build in its build script and fails with "Could not find `protoc`" long
# after everything else has succeeded. clang is there for bindgen, which pjrt-sys
# also runs; binutils gives us nm, used below to prove libtpu really is a PJRT
# plugin rather than merely a file that exists.
apt-get install -y \
  build-essential clang cmake pkg-config protobuf-compiler binutils \
  curl git ca-certificates python3 python3-pip

# rustup rather than the distro rustc: Ubuntu 22.04 ships 1.75, and the rlx
# crates need a 2024-edition compiler. Pinned, because "whatever is newest today"
# is not a thing a capacity cycle should depend on.
if ! [ -x "$CARGO_HOME/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain {rust_toolchain} --profile minimal
fi
export PATH="$CARGO_HOME/bin:$PATH"
rustc --version
cargo --version

# libtpu ships on the JAX release index, not PyPI. We want the shared object and
# nothing else — no JAX, no jaxlib, no Python runtime dependency in the serving
# path. --no-deps keeps it to exactly that.
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade --no-deps {libtpu_spec} \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Find the plugin the wheel just dropped. Hardcoding a site-packages path breaks
# whenever the interpreter version moves, which on these images it does.
LIBTPU_SO=$(find /usr/local/lib /usr/lib/python3 -name libtpu.so -print -quit 2>/dev/null || true)
if [ -z "$LIBTPU_SO" ]; then
  echo "ERROR: libtpu installed but no libtpu.so found on disk"
  exit 1
fi

# A file called libtpu.so is not yet a PJRT plugin. The one thing that makes it
# one is the GetPjrtApi entry point, and checking for it here costs nothing and
# turns an obscure runtime dlsym failure into a boot-time error with a name.
if ! nm -D --defined-only "$LIBTPU_SO" | grep -q GetPjrtApi; then
  echo "ERROR: $LIBTPU_SO exports no GetPjrtApi — not a PJRT plugin"
  exit 1
fi

# Both names are in circulation: LIBTPU_PATH is what rlx-tpu reads and
# TPU_LIBRARY_PATH is what JAX reads. Set both, so a later comparison against the
# Python path on the same VM does not turn into an argument about which plugin
# each side loaded.
mkdir -p /etc/profile.d
cat > /etc/profile.d/jaxrust.sh <<PROFILE
export CARGO_HOME=/opt/rust/cargo
export RUSTUP_HOME=/opt/rust/rustup
export PATH="/opt/rust/cargo/bin:\$PATH"
export LIBTPU_PATH="$LIBTPU_SO"
export TPU_LIBRARY_PATH="$LIBTPU_SO"
PROFILE
chmod 0644 /etc/profile.d/jaxrust.sh

# The toolchain is installed as root but built against by the SSH user, so the
# cargo/rustup trees have to be readable and the registry cache writable.
chmod -R a+rX /opt/rust
mkdir -p /opt/rust/cargo/registry
chmod -R 1777 /opt/rust/cargo/registry

# libtpu creates /tmp/tpu_logs as root here; without this every later non-root
# run spams "Could not open the log file ... Permission denied".
mkdir -p /tmp/tpu_logs
chmod -R 1777 /tmp/tpu_logs 2>/dev/null || true

echo "libtpu: $LIBTPU_SO"
echo "JAXRUST-BOOTLOADER: TPU environment ready."
