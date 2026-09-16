#!/usr/bin/env bash
# Thread + CPU-affinity lever sweep for local-llamacpp-cpu-2b-q4_0.
#
# llama-bench, NOT sweep.py: thread count and CPU mask are server LAUNCH flags,
# so sweeping them over HTTP needs a server restart per cell. llama-bench
# isolates the lever. The cost is that these are engine numbers with no HTTP
# path — never difference them against a serving report.
#
# PORTABLE ON PURPOSE. Nothing here hardcodes a mask or a thread count: both are
# derived from sysfs by topology.py, because a hex mask is a fact about one die
# and not configuration. The first version of this script carried this host's
# five masks as constants, which would have pinned to arbitrary cores on any
# other CPU and returned numbers that looked fine and meant nothing.
#
# Stop llama-server first so it is not holding ~4.2 GB and 4 threads beside the
# benchmark. Results go to results/ and the derived topology is recorded there
# so two hosts' runs can be told apart and compared.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here/../../.."
. ./tpu.env

B="$LLAMA_CPP_DIR/build-cpu/bin/llama-bench"
[ -x "$B" ] || { echo "no llama-bench at $B — run 'make build' first" >&2; exit 1; }
[ -f "$MODEL_PATH" ] || { echo "no GGUF at $MODEL_PATH — run 'make download'" >&2; exit 1; }

mkdir -p "$here/results"
eval "$(python3 "$here/topology.py" --shell)"
python3 "$here/topology.py" --json > "$here/results/topology.json"
echo "host topology: hybrid=$TOPO_HYBRID smt=$TOPO_SMT perf=[$TOPO_PERF_CPUS] eff=[$TOPO_EFF_CPUS]"

# CUDA_VISIBLE_DEVICES= is load-bearing on a host that HAS a GPU, which the
# control box does: -ngl 0 alone still lets a CUDA build initialise the device
# and offload large prompt batches. The binary is from build-cpu (GGML_CUDA=OFF),
# so this is the second of two guards, not the only one.
common=(-m "$MODEL_PATH" -ngl 0 -ctk "$KV_CACHE_TYPE" -ctv "$KV_CACHE_TYPE" \
        -fa on -p 512 -n 128)
run() { CUDA_VISIBLE_DEVICES= "$B" "${common[@]}" "$@" -o md 2>&1 | grep -E '^\| (model|gemma4)'; }

# ── Pass 1: thread count, unpinned ───────────────────────────────────────────
# Derived, not chosen: the decode set, the prefill set, perf+eff, and every
# online CPU. On an i7-1360P that is 4,8,12,16 — the 2026-09-16 sweep exactly.
threads=$(printf '%s\n' "$TOPO_THREADS_DECODE" "$TOPO_THREADS_PREFILL" \
                        "$TOPO_THREADS_PERF_PLUS_EFF" \
                        "$(echo "$TOPO_ONLINE_CPUS" | tr ',' '\n' | wc -l)" \
          | sort -n -u | paste -sd,)
echo "=== pass 1: -t $threads unpinned ==="
run -t "$threads" -r 3 | tee "$here/results/llama-bench-threads.md"

# ── Pass 2: affinity ─────────────────────────────────────────────────────────
# Cells whose mask is empty are not expressible on this host (no hybrid split,
# or no SMT) and are RECORDED AS INFEASIBLE rather than silently dropped.
: > "$here/results/llama-bench-affinity.md"
cell() {  # label, threads, mask
  local label="$1" t="$2" m="$3"
  { echo "### -t $t -C $m  ($label)"
    if [ -z "$m" ] || [ "$t" -eq 0 ]; then
      echo "_infeasible on this host: topology cannot express this cell_"
    else
      run -t "$t" -C "$m" --cpu-strict 1 -r 5 | grep -E '^\| gemma4'
    fi
  } >> "$here/results/llama-bench-affinity.md"
}

cell "perf-cores-one-thread-each-no-SMT" "$TOPO_THREADS_DECODE"       "$TOPO_MASK_DECODE"
cell "perf-cores-SMT-doubled"            "$TOPO_THREADS_PREFILL"      "$TOPO_MASK_PREFILL"
cell "eff-cores-only"                    "$TOPO_THREADS_EFF_ONLY"     "$TOPO_MASK_EFF_ONLY"
cell "perf-distinct-plus-all-eff"        "$TOPO_THREADS_PERF_PLUS_EFF" "$TOPO_MASK_PERF_PLUS_EFF"

cat "$here/results/llama-bench-affinity.md"
echo
echo "NOTE: the 2026-09-16 run also has a '2-P-both-SMT-siblings' (0x0F) cell."
echo "It is CONFOUNDED — half the perf cores AND SMT doubling — and is not"
echo "reproduced here. See REPORT.md; cite prefill-vs-decode SMT from the"
echo "perf-cores pair above instead."
