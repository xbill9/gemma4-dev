#!/usr/bin/env python3
"""Republish $LATEST with no environments running, then start exactly MIN_ENVS/MAX_ENVS.

A republish starts the new environments before it stops the old ones. With the capacity provider at its
vCPU cap (the quota here allows two g6.2xlarge) that fails, and Lambda then rolls $LATEST.PUBLISHED back to
the previous code and configuration ("reached maxVCpuCount limit", 2026-11-18). This scales to 0, publishes,
scales back up, and waits for the new environment's init report. It prints the engine's start-up lines (ENGINE_LOG_PATTERN) so the
caller can check the configuration it actually started with.

    republish.py
"""

import json
import sys
import time

import lmi

E = lmi.E
FN = E["FUNCTION_NAME"]
Q = "$LATEST.PUBLISHED"


def cfg():
    return lmi.aws("lambda", "get-function-configuration", "--function-name", f"{FN}:{Q}")[0]


def wait(pred, what, every=15, limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        c = cfg()
        if pred(c):
            print(
                f"{time.strftime('%H:%M:%S')} {what}: {c.get('State')} {c.get('LastUpdateStatus')} "
                f"after {time.time() - t0:.0f} s",
                flush=True,
            )
            return c
        if c.get("LastUpdateStatus") == "Failed":
            sys.exit(f"update failed: {c.get('LastUpdateStatusReason')}")
        time.sleep(every)
    sys.exit(f"timed out waiting for {what}")


def main():
    latest = lmi.aws("lambda", "get-function-configuration", "--function-name", FN)[0]
    lmi.scaling(0, 0)
    wait(lambda c: c.get("State") == "Deactivated" and c.get("LastUpdateStatus") != "InProgress", "scaled to 0")
    t_pub = time.time()
    v = lmi.aws("lambda", "publish-version", "--function-name", FN, "--publish-to", "LATEST_PUBLISHED")[0]
    print("published", v.get("Version"), v.get("CodeSha256"), flush=True)
    wait(
        lambda c: c.get("CodeSha256") == latest["CodeSha256"] and c.get("LastUpdateStatus") != "InProgress",
        "published code in place",
    )
    t_up = time.time()
    lmi.scaling(E["MIN_ENVS"], E["MAX_ENVS"])
    wait(lambda c: c.get("State") == "Active" and c.get("LastUpdateStatus") != "InProgress", "active")
    # The init report and engine line of the environment that started after scaling up.
    for _ in range(40):
        ev = lmi.aws(
            "logs",
            "filter-log-events",
            "--log-group-name",
            f"/aws/lambda/{FN}",
            "--start-time",
            str(int(t_up * 1000)),
            "--filter-pattern",
            f'?"platform.initReport" {E.get("ENGINE_LOG_PATTERN", "")}'.strip(),
        )[0]
        msgs = [(e["logStreamName"], e["message"]) for e in ev.get("events", [])]
        if any("initReport" in m for _, m in msgs):
            for s, m in msgs:
                print(s.split("]")[-1][:12], m[:400], flush=True)
            break
        time.sleep(15)
    c = cfg()
    print(
        json.dumps(
            {
                "code": c.get("CodeSha256"),
                "args": {k: v for k, v in c["Environment"]["Variables"].items() if k.endswith(("_ARGS", "_CMD"))},
                "publish_to_code_s": round(t_up - t_pub, 1),
                "scale_up_to_active_s": round(time.time() - t_up, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
