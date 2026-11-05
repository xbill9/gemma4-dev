"""Build and run the `mistralrs serve` command for this rig from tpu.env.

    python3 serve.py            # exec mistralrs serve in the foreground
    python3 serve.py --print    # print the command and exit

Values come from tpu.env; a real environment variable wins over the file. One builder, so
`make serve` and every benchmark script start the same server.
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
        env["MISTRALRS_BIN"],
        "serve",
        "-f",
        env["MODEL_PATH"],
        "--mmproj",
        env["MMPROJ_PATH"],
        "--host",
        env["HOST"],
        "-p",
        env["PORT"],
        "--max-seqs",
        env["MAX_SEQS"],
        "--max-model-len",
        env["MAX_MODEL_LEN"],
        "--prefix-cache-n",
        env["PREFIX_CACHE_N"],
        "--kv-cache",
        env["KV_CACHE"],
        "--max-num-batched-tokens",
        env["MAX_NUM_BATCHED_TOKENS"],
        "--no-ui",
    ]
    if env.get("MTP_MODEL"):
        cmd += ["--mtp-model", env["MTP_MODEL"]]
        if env.get("MTP_N_PREDICT"):
            cmd += ["--mtp-n-predict", env["MTP_N_PREDICT"]]
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
