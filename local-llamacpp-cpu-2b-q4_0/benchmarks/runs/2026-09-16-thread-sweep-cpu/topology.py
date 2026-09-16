"""Derive this host's CPU topology and llama.cpp affinity masks from sysfs.

A hex mask is a fact about one die, never configuration: 0x55 means "one thread
per performance core" only on a host whose P-cores are logical 0-7 with SMT. So
the sweep derives its masks here rather than carrying constants, and dumps what
it derived beside the results so two hosts' runs can be compared.

Emits shell assignments with --shell, or a JSON topology record with --json.
"""
import json
import sys


def _parse_cpu_list(text):
    out = set()
    for part in (text or "").strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def _read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def online_cpus():
    return _parse_cpu_list(_read("/sys/devices/system/cpu/online")) or {0}


def perf_cpus():
    """Logical CPUs on the performance cores; all online CPUs if not hybrid."""
    hybrid = _read("/sys/devices/cpu_core/cpus")
    return _parse_cpu_list(hybrid) if hybrid else online_cpus()


def eff_cpus():
    """Logical CPUs on the efficiency cores; empty if not hybrid."""
    return _parse_cpu_list(_read("/sys/devices/cpu_atom/cpus"))


def sibling_groups(cpus):
    groups, seen = [], set()
    for c in sorted(cpus):
        sib = _read(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list")
        g = frozenset(_parse_cpu_list(sib) & cpus) if sib else frozenset({c})
        if g and g not in seen:
            seen.add(g)
            groups.append(g)
    return groups


def one_per_physical(cpus):
    return {min(g) for g in sibling_groups(cpus)}


def mask(cpus):
    return "0x%X" % sum(1 << c for c in cpus)


def topology():
    perf, eff = perf_cpus(), eff_cpus()
    groups = sibling_groups(perf)
    decode = one_per_physical(perf)
    smt = any(len(g) > 1 for g in groups)
    return {
        "online_cpus": sorted(online_cpus()),
        "hybrid": bool(eff),
        "smt": smt,
        "perf_cpus": sorted(perf),
        "eff_cpus": sorted(eff),
        "perf_physical_cores": len(groups),
        # Prefill: every perf logical thread. SMT measured neutral-to-good here.
        "mask_prefill": mask(perf),
        "threads_prefill": len(perf),
        # Decode: one thread per physical perf core. SMT measured -5.6%.
        "mask_decode": mask(decode),
        "threads_decode": len(decode),
        # Diagnostic cells. Empty string where the host cannot express them.
        "mask_eff_only": mask(eff) if eff else "",
        "threads_eff_only": len(eff),
        "mask_perf_plus_eff": mask(decode | eff) if eff else "",
        "threads_perf_plus_eff": len(decode | eff),
    }


if __name__ == "__main__":
    t = topology()
    if "--json" in sys.argv:
        print(json.dumps(t, indent=2))
    else:
        for k, v in t.items():
            if isinstance(v, list):
                v = ",".join(str(x) for x in v)
            if isinstance(v, bool):
                v = int(v)
            print(f"TOPO_{k.upper()}={v}")
