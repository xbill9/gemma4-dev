"""Build and run the `sglang serve` command for this rig from tpu.env.

    python3 serve.py            # exec the server in the foreground
    python3 serve.py --print    # print the command and exit

Values come from tpu.env; a real environment variable wins over the file.
"""

import os
import shlex
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env(path: str = os.path.join(HERE, "tpu.env")) -> Dict[str, str]:
    """KEY=VALUE lines of tpu.env, with os.environ taking precedence."""
    vals: Dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            vals[key.strip()] = value.strip().strip('"')
    for key in list(vals):
        if key in os.environ:
            vals[key] = os.environ[key]
    return vals


def command(env: Dict[str, str]) -> List[str]:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        env["MODEL_NAME"],
        "--revision",
        env["MODEL_REVISION"],
        "--host",
        env["HOST"],
        "--port",
        env["PORT"],
        "--mem-fraction-static",
        env["MEM_FRACTION_STATIC"],
        "--context-length",
        env["CONTEXT_LENGTH"],
        "--max-running-requests",
        env["MAX_RUNNING_REQUESTS"],
    ]
    if env.get("TOOL_CALL_PARSER"):
        cmd += ["--tool-call-parser", env["TOOL_CALL_PARSER"]]
    if env.get("REASONING_PARSER"):
        cmd += ["--reasoning-parser", env["REASONING_PARSER"]]
    if env.get("DISABLE_RADIX_CACHE", "").lower() == "true":
        cmd.append("--disable-radix-cache")
    if env.get("SPECULATIVE_MODEL"):
        steps = int(env["SPECULATIVE_NUM_STEPS"])
        cmd += ["--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", env["SPECULATIVE_MODEL"],
                "--speculative-num-steps", str(steps), "--speculative-eagle-topk", "1",
                "--speculative-num-draft-tokens", str(steps + 1)]
    return cmd


def main() -> int:
    cmd = command(load_env())
    if "--print" in sys.argv[1:]:
        print(shlex.join(cmd))
        return 0
    os.execvp(cmd[0], cmd)
    return 1


if __name__ == "__main__":
    sys.exit(main())
