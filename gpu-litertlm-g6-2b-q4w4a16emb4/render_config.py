"""Render LiteRT-LM's config.json for this rig from tpu.env.

`litert-lm serve` takes no per-model flags; backend, cache and speculative
decoding come from a config.json keyed by model id. This writes one from
tpu.env (a real environment variable wins) so `make serve` never touches
~/.litert-lm/config.json.

    python3 render_config.py [OUT]       # default run/config.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

HERE = Path(__file__).resolve().parent

SPEC_VALUES = {"on", "off", "adaptive"}
BACKEND_VALUES = {"cpu", "gpu", "npu"}
CACHE_VALUES = {"disk", "memory", "no"}


def read_env(path: Path) -> Dict[str, str]:
    """Parse KEY=VALUE lines; strip one level of surrounding quotes."""
    values: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def setting(env: Dict[str, str], key: str) -> Optional[str]:
    value = os.environ.get(key, env.get(key, ""))
    return value or None


def build_config(env: Dict[str, str]) -> dict:
    model_id = setting(env, "MODEL_ID")
    if not model_id:
        raise ValueError("MODEL_ID is not set")
    cfg: dict = {}
    for key, field, allowed in (
        ("BACKEND", "backend", BACKEND_VALUES),
        ("CACHE", "cache", CACHE_VALUES),
        ("SPECULATIVE_DECODING", "speculative_decoding", SPEC_VALUES),
    ):
        value = setting(env, key)
        if value is None:
            continue
        if value not in allowed:
            raise ValueError(f"{key}={value!r}; expected one of {sorted(allowed)}")
        cfg[field] = value
    max_tokens = setting(env, "MAX_NUM_TOKENS")
    if max_tokens is not None:
        cfg["max_num_tokens"] = int(max_tokens)
    return {"models": {model_id: cfg}}


def main(argv: list) -> int:
    out = Path(argv[1]) if len(argv) > 1 else HERE / "run" / "config.json"
    config = build_config(read_env(HERE / "tpu.env"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {out}: {json.dumps(config)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
