"""Read Bespoke Labs' 13-subset public suite through one arm and record the answer distributions.

    python3 nimble_suite/run_suite.py --arm autoregressive --upstream http://localhost:8000 \
        --model cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit --records <workdir>/public --run RUN

Each record's `input` is a Jev request (`state` plus one typed question). The PR's
proxy turns it into a schema with its own Jev parser (`jev_schema`, `jev_state`),
so both arms see the proxy's system prompt, answer template and single-token
labels, and both read label probabilities the same way as run_eval.py:

  autoregressive  the served model's own chat template, then the answer lead,
                  one completion token, label logprobs
  diffusion       the proxy's one-step read-only denoise over the seeded canvas,
                  `--reads` noise draws with the proxy's seed schedule

Probabilities are stored under the keys Bespoke Labs' scorer uses: "true"/"false"
for noul, option names for choice, "0".."N-1" for score levels.

Results land in results/RUN/<arm>-<subset>.jsonl; reruns skip ids already written.
"""

import argparse
import glob
import json
import os
import sys
import time
import types
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "vendor"))

import structured_server as ss  # noqa: E402

SEED_STRIDE = 7919
MODEL_TOK = None


def keys_for(q):
    if q["type"] == "noul":
        return ["true", "false"]  # the proxy's noul labels are yes, no in that order
    if q["type"] == "score":
        return [str(i) for i in range(len(q["choices"]))]
    return [c[0] for c in q["choices"]]


def own_prompt(sys_text, state, template, slot):
    out = MODEL_TOK.apply_chat_template(
        [{"role": "system", "content": sys_text}, {"role": "user", "content": state}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = [int(t) for t in (out["input_ids"] if hasattr(out, "keys") else out)]
    return ids + template[len(ss.SCAFFOLD): slot["pos"]]


def read_ar(schema, template, slots, sys_text, state):
    body = {
        "model": ss.ARGS.model,
        "prompt": own_prompt(sys_text, state, template, slots[0]),
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": ss.TOPK,
        "logprob_token_ids": ss.label_id_union(slots),
        "return_tokens_as_token_ids": True,
    }
    t0 = time.time()
    d = ss.upstream_completions(body)
    ms = (time.time() - t0) * 1e3
    row = d["choices"][0]["logprobs"]["top_logprobs"][0]
    top = {int(k.split(":")[1]): v for k, v in row.items()}
    return [ss.slot_distribution(top, slots[0]["label_ids"])], {"first_ms": ms, "extra_ms": None}


def read_dg(schema, template, slots, sys_text, state, seed, reads):
    t0 = time.time()
    first, _ = ss.one_read(schema, template, slots, sys_text, state, seed)
    first_ms = (time.time() - t0) * 1e3
    out = [first[0]]
    extra_ms = None
    if reads > 1:
        t1 = time.time()
        with ThreadPoolExecutor(reads - 1) as pool:
            out += list(pool.map(lambda k: ss.one_read(schema, template, slots, sys_text, state, seed + k * SEED_STRIDE)[0][0], range(1, reads)))
        extra_ms = (time.time() - t1) * 1e3
    return out, {"first_ms": first_ms, "extra_ms": extra_ms}


def main():
    global MODEL_TOK
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["autoregressive", "diffusion"], required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--records", required=True, help="<workdir>/public")
    ap.add_argument("--run", required=True)
    ap.add_argument("--reads", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    ss.ARGS = types.SimpleNamespace(model=args.model, upstream=args.upstream)
    ss.init_tokenizer(AutoTokenizer.from_pretrained(args.tokenizer))
    if args.arm == "autoregressive":
        MODEL_TOK = AutoTokenizer.from_pretrained(args.model)
    outdir = os.path.join(HERE, "results", args.run)
    os.makedirs(outdir, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(args.records, "*", "all.jsonl"))):
        sub = os.path.basename(os.path.dirname(path))
        if sub == "hs2full":
            continue
        records = [json.loads(line) for line in open(path)]
        if args.limit:
            records = records[: args.limit]
        # One file per arm and subset: a second model in the same run would resume
        # the first model's file and skip every record, so each model needs its own run.
        out_path = os.path.join(outdir, f"{args.arm}-{sub}.jsonl")
        if os.path.exists(out_path):
            first = json.loads(open(out_path).readline() or "{}")
            if first and first.get("model") != args.model:
                raise SystemExit(f"{out_path} holds {first.get('model')}; use a separate --run for {args.model}")
        done = set()
        if os.path.exists(out_path):
            done = {json.loads(line)["id"] for line in open(out_path)}
        todo = [r for r in records if r["id"] not in done]

        def one(rec):
            body = {"state": rec["input"]["state"], "questions": rec["input"]["questions"]}
            schema = ss.jev_schema(body)
            state = ss.jev_state(body)
            template, slots = ss.template_for(schema, ss.SCAFFOLD, "")
            sys_text = ss.system_text(schema)
            q = schema["questions"][0]
            if args.arm == "autoregressive":
                reads, lat = read_ar(schema, template, slots, sys_text, state)
            else:
                reads, lat = read_dg(schema, template, slots, sys_text, state, zlib.crc32(rec["id"].encode()), args.reads)
            return {
                "id": rec["id"],
                "subset": sub,
                "family": rec.get("family"),
                "type": q["type"],
                "keys": keys_for(q),
                "target": rec["reference"]["target"],
                "distribution": rec["reference"].get("distribution"),
                "arm": args.arm,
                "model": args.model,
                "reads": reads,
                "latency": lat,
            }

        t0 = time.time()
        with open(out_path, "a") as f, ThreadPoolExecutor(args.concurrency) as pool:
            for i, row in enumerate(pool.map(one, todo), 1):
                f.write(json.dumps(row) + "\n")
                f.flush()
                if i % 100 == 0 or i == len(todo):
                    print(f"{args.arm} {sub}: {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
