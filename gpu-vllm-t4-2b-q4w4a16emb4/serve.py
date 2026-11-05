"""Build and run the `vllm serve` command for this rig from tpu.env.

    python3 serve.py            # exec vllm serve in the foreground
    python3 serve.py --print    # print the command and exit

Values come from tpu.env; a real environment variable wins over the file. One builder, so
`make serve` and every benchmark script start the same server.
"""

import json
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
        env.get("PYTHON_BIN", "python3"),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        env["MODEL_NAME"],
        "--revision",
        env["MODEL_REVISION"],
        "--host",
        env["HOST"],
        "--port",
        env["PORT"],
        "--gpu-memory-utilization",
        env["GPU_MEMORY_UTILIZATION"],
        "--max-model-len",
        env["MAX_MODEL_LEN"],
        "--max-num-seqs",
        env["MAX_NUM_SEQS"],
        "--tensor-parallel-size",
        env["TENSOR_PARALLEL_SIZE"],
    ]
    if env.get("DTYPE"):
        cmd += ["--dtype", env["DTYPE"]]
    if env.get("LANGUAGE_MODEL_ONLY", "").lower() == "true":
        cmd.append("--language-model-only")
    if env.get("TOOL_CALL_PARSER"):
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", env["TOOL_CALL_PARSER"]]
    if env.get("REASONING_PARSER"):
        cmd += ["--reasoning-parser", env["REASONING_PARSER"]]
    if env.get("SPECULATIVE_MODEL"):
        spec = {"model": env["SPECULATIVE_MODEL"], "num_speculative_tokens": int(env["NUM_SPECULATIVE_TOKENS"])}
        cmd += ["--speculative-config", json.dumps(spec)]
    return cmd


def main() -> int:
    cmd = command(load_env())
    if "--print" in sys.argv[1:]:
        print(shlex.join(cmd))
        return 0
    env = load_env()
    for key in ("PYTHONUSERBASE", "PIP_CACHE_DIR", "TMPDIR"):
        if env.get(key):
            os.environ[key] = env[key]
    os.execvp(cmd[0], cmd)
    return 1


if __name__ == "__main__":
    sys.exit(main())
