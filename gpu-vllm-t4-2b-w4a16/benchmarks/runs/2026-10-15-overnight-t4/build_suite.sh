#!/bin/bash
# Rebuild the Nimble public suite on this host (it has never been built here) with
# jev-tpu-v5e1/nimble_suite/build.sh, into /opt1/data/nimble-suite. CPU and network only; runs
# beside the GPU quality job, whose scores do not depend on host CPU. pyarrow goes into the
# python3.13 user site on /opt1 (no virtualenv); the nimble converters run on that interpreter.
set -eux
export PYTHONUSERBASE=/opt1/pyuser TMPDIR=/opt1/tmp PIP_CACHE_DIR=/opt1/pipcache
/usr/bin/python3.13 -m pip install --user --break-system-packages pyarrow
mkdir -p /opt1/data/nimble-suite /opt1/data/bin
ln -sf /usr/bin/python3.13 /opt1/data/bin/python3
PATH=/opt1/data/bin:$PATH bash $HOME/gemma4-dev/jev-tpu-v5e1/nimble_suite/build.sh /opt1/data/nimble-suite
ls -la /opt1/data/nimble-suite/public && touch /opt1/data/nimble-suite/BUILD_OK
