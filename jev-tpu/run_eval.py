"""Read every labelled example through one arm and record the label distributions.

    # DiffusionGemma, one denoise step over the seeded canvas, 4 noise draws
    python3 run_eval.py --arm diffusion --upstream http://127.0.0.1:8000 \
        --model google/diffusiongemma-26B-A4B-it --run RUN

    # Gemma 4 26B read left to right: next-token logprobs at the same slot
    python3 run_eval.py --arm autoregressive --upstream http://127.0.0.1:8001 \
        --model google/gemma-4-26B-A4B-it --run RUN

Both arms use vLLM PR #57250's structured_server.py (vendored at the merge
commit) for everything but the read itself: the system prompt, the answer
template, the single-token label check, the label token ids, and
`slot_distribution`, which turns returned logprobs into label probabilities,
label mass and entropy. What differs is only how the answer slot is read:

  diffusion       the proxy's own `one_read`: the canvas holds the template with
                  a random vocabulary token in the slot, one denoise step,
                  read-only, logprobs at the slot. Repeated with the proxy's
                  seed schedule, so read 0 is a single read and reads 0..3 are
                  what `samples: "auto"` averages when it escalates.
  autoregressive  the identical token prefix (chat prompt, empty thought block,
                  "id: ") sent as a completion for one token, logprobs at that
                  next position. Deterministic, so one read.

Results land in results/RUN/<arm>-<task>.jsonl, one line per example. A rerun
skips ids already written, so an interrupted run resumes.
"""

import argparse
import json
import os
import sys
import time
import types
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor"))

import structured_server as ss  # noqa: E402
from tasks import TASKS  # noqa: E402

SEED_STRIDE = 7919  # structured_server.read_many's schedule: seed + k * 7919


def example_seed(ex_id):
    return zlib.crc32(ex_id.encode())


def setup(task_name, variant="none"):
    """variant "reversed" lists a choice question's options in reverse order, so
    each option gets a different letter; comparing predicted option names with
    the unreversed run measures sensitivity to option order. Yes/no questions
    have no order to reverse and are left unchanged."""
    task = TASKS[task_name]
    question = dict(task["question"])
    if variant == "reversed" and question["type"] == "choice":
        question["options"] = list(reversed(question["options"]))
    schema = ss.parse_schema({"questions": [question], "samples": 1})
    template, slots = ss.template_for(schema, ss.SCAFFOLD, "")
    q = schema["questions"][0]
    names = [c[0] for c in q["choices"]]
    return schema, template, slots, q, names


def state_text(ex):
    return json.dumps({"text": ex["text"]})


def diffusion_record(schema, template, slots, sys_text, ex, reads):
    state = state_text(ex)
    seed = example_seed(ex["id"])
    out = []
    t0 = time.time()
    first, _ = ss.one_read(schema, template, slots, sys_text, state, seed)
    first_ms = (time.time() - t0) * 1e3
    out.append(first[0])
    extra_ms = None
    if reads > 1:
        # the proxy's escalation fires the remaining reads in parallel
        t1 = time.time()
        with ThreadPoolExecutor(reads - 1) as pool:
            rest = list(
                pool.map(
                    lambda k: ss.one_read(
                        schema, template, slots, sys_text, state, seed + k * SEED_STRIDE
                    )[0][0],
                    range(1, reads),
                )
            )
        extra_ms = (time.time() - t1) * 1e3
        out += rest
    return out, {"first_ms": first_ms, "extra_ms": extra_ms}


MODEL_TOK = None  # the served model's own tokenizer, set in main for the plain arm


def own_template_ids(tok, sys_text, state):
    out = tok.apply_chat_template(
        [{"role": "system", "content": sys_text}, {"role": "user", "content": state}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return [int(t) for t in (out["input_ids"] if hasattr(out, "keys") else out)]


def ar_prompt(sys_text, ex, template, slot):
    """The served model's own chat template, then the answer lead up to the
    slot. For Gemma 4 26B the template already ends with the empty thought
    block, which makes this the same token sequence as DiffusionGemma's prompt
    plus the canvas template; check_prefix confirms that. Gemma 4 E2B and E4B
    templates end at the model turn with no thought block, and get none."""
    own = own_template_ids(MODEL_TOK, sys_text, state_text(ex))
    return own + template[len(ss.SCAFFOLD): slot["pos"]]


def autoregressive_record(template, slots, sys_text, ex):
    slot = slots[0]
    body = {
        "model": ss.ARGS.model,
        "prompt": ar_prompt(sys_text, ex, template, slot),
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
    return [ss.slot_distribution(top, slot["label_ids"])], {
        "first_ms": ms,
        "extra_ms": None,
    }


def check_prefix(model):
    """Loads the served model's tokenizer for ar_prompt, and checks it against
    DiffusionGemma's prompt. The only difference allowed is the empty thought
    block at the end of the chat template; anything else aborts the run."""
    global MODEL_TOK
    from transformers import AutoTokenizer

    MODEL_TOK = AutoTokenizer.from_pretrained(model)
    notes = set()
    for name in TASKS:
        schema, template, slots, _, _ = setup(name)
        sys_text = ss.system_text(schema)
        ex = {"text": "check"}
        dg = ss.chat_prompt_ids(sys_text, state_text(ex)) + template[: slots[0]["pos"]]
        mine = ar_prompt(sys_text, ex, template, slots[0])
        if mine == dg:
            notes.add("identical to the DiffusionGemma prompt")
        elif mine == dg[: len(dg) - len(template[: slots[0]["pos"]])] + template[len(ss.SCAFFOLD): slots[0]["pos"]]:
            notes.add("DiffusionGemma prompt without the empty thought block (the model's own template has none)")
        else:
            raise SystemExit(f"{name}: {model}'s chat template renders a different prefix")
    print(f"prompt check for {model}: " + "; ".join(sorted(notes)), flush=True)


def returned(dist, label_ids, keep_top=0):
    """How many label ids came back with a real logprob. A label missing from
    the response is scored at slot_distribution's floor, which is a guess.
    With keep_top, the highest-scoring returned tokens are kept as
    [token_id, logprob] pairs, to show where probability off the labels goes."""
    top = dist.pop("_top", None)
    if top is not None and keep_top:
        dist["top"] = sorted(([int(k), v] for k, v in top.items()), key=lambda kv: -kv[1])[:keep_top]
    return None if top is None else sum(i in top for i in label_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["diffusion", "autoregressive"], required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="google/diffusiongemma-26B-A4B-it")
    ap.add_argument("--run", required=True, help="results/<run>/")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS))
    ap.add_argument("--reads", type=int, default=4, help="diffusion noise draws")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N examples only")
    ap.add_argument("--variant", choices=["none", "reversed"], default="none")
    ap.add_argument("--keep-top", type=int, default=0, help="store the N highest returned tokens per read")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    ss.ARGS = types.SimpleNamespace(model=args.model, upstream=args.upstream)
    ss.init_tokenizer(AutoTokenizer.from_pretrained(args.tokenizer))
    # keep the raw returned logprobs beside each distribution, so the record can
    # say whether every label came back or was scored at the floor
    orig = ss.slot_distribution
    ss.slot_distribution = lambda top, label_ids: orig(top, label_ids) | {"_top": top}
    if args.arm == "autoregressive":
        check_prefix(args.model)

    outdir = os.path.join(HERE, "results", args.run)
    os.makedirs(outdir, exist_ok=True)
    for name in args.tasks:
        if args.variant == "reversed" and TASKS[name]["question"]["type"] != "choice":
            continue
        schema, template, slots, q, names = setup(name, args.variant)
        sys_text = ss.system_text(schema)
        with open(os.path.join(HERE, "data", f"{name}.jsonl")) as f:
            examples = [json.loads(line) for line in f]
        if args.limit:
            examples = examples[: args.limit]
        suffix = "" if args.variant == "none" else f"--{args.variant}"
        path = os.path.join(outdir, f"{args.arm}-{name}{suffix}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {json.loads(line)["id"] for line in f}
        todo = [ex for ex in examples if ex["id"] not in done]

        def one(ex):
            if args.arm == "diffusion":
                reads, lat = diffusion_record(schema, template, slots, sys_text, ex, args.reads)
            else:
                reads, lat = autoregressive_record(template, slots, sys_text, ex)
            for r in reads:
                r["labels_returned"] = returned(r, slots[0]["label_ids"], args.keep_top)
            return {
                "id": ex["id"],
                "task": name,
                "arm": args.arm,
                "variant": args.variant,
                "model": args.model,
                "gold": ex["gold"],
                "names": names,
                "labels": q["labels"],
                "reads": reads,
                "latency": lat,
            }

        t0 = time.time()
        with open(path, "a") as f, ThreadPoolExecutor(args.concurrency) as pool:
            for i, rec in enumerate(pool.map(one, todo), 1):
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if i % 50 == 0 or i == len(todo):
                    print(f"{args.arm} {name}: {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        if not todo:
            print(f"{args.arm} {name}: already complete ({len(done)})")


if __name__ == "__main__":
    main()
