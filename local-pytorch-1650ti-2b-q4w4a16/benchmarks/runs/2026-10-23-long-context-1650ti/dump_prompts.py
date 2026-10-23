"""Rebuild the exact prompts e2b_long.py used (same seed, same order of depths) and save them for other engines."""
import json, random, sys
from transformers import AutoTokenizer
TEXT = "xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text"
tok = AutoTokenizer.from_pretrained(TEXT)
rng = random.Random(1234)
QUESTION = "\n\nQuestion: what is the code for record {r:06d}? Answer with the code only."
out = []
for depth in [int(x) for x in sys.argv[1].split(",")]:
    head = tok.apply_chat_template([{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False)
    overhead = len(tok(head + QUESTION.format(r=0), add_special_tokens=False)["input_ids"])
    records, used, codes, i = [], overhead, {}, 0
    while True:
        i += 1
        code = rng.randrange(100000, 999999)
        line = f"Record {i:06d}: the code for record {i:06d} is {code}.\n"
        cost = len(tok(line, add_special_tokens=False)["input_ids"])
        if used + cost > depth and records:
            break
        records.append(line); codes[i] = code; used += cost
        if depth == 0:
            break
    target = max(1, len(records) // 4)
    content = "".join(records) + QUESTION.format(r=target)
    by_code = {c: r for r, c in codes.items()}
    out.append({"depth": depth, "records": len(records), "target": target, "code": codes[target], "content": content,
                "code_owner": by_code})
json.dump([{k: v for k, v in o.items() if k != "code_owner"} for o in out], open(sys.argv[2], "w"))
for o in out:
    wrong = int(sys.argv[3]) if len(sys.argv) > 3 else None
    print(json.dumps({"depth": o["depth"], "records": o["records"], "target": o["target"], "code": o["code"],
                      "wrong_answer_is_code_of_record": o["code_owner"].get(wrong)}))
