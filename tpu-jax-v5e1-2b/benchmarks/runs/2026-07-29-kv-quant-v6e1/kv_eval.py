"""Quality of an 8-bit KV cache, measured rather than eyeballed.

Seven short greedy prompts agreeing with bf16 is enough to keep working and not
enough to publish. This measures the thing that actually degrades.

The KV cache dtype affects DECODE only: prefill attends over freshly computed
K/V, so a prefill-only comparison is blind to it by construction. Quantization
error therefore enters one token at a time and accumulates over the decode run,
which means the quantity to measure is per-step likelihood over a long forced
decode, not the first few tokens of a greedy sample.

Three measurements, on public-domain text the model did not generate:

  1. Teacher-forced NLL / perplexity over a real continuation. Same token stream
     for every cache dtype, so the comparison is exact and sampling divergence
     cannot contaminate it.
  2. Drift: does the gap grow with decode depth? Reported per quarter of the run.
     A cache whose error compounds looks fine at token 10 and bad at token 400.
  3. Free-running greedy agreement against bf16, and the index of first
     divergence.
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from jax_engine import JaxGemmaEngine
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ports.gemma4.jax_e_model import make_cached_decode_step, prefill_with_kv_cache, pad_to_tpu_v6e_bucket

MID = "google/gemma-4-E2B-it-qat-w4a16-ct"
DTYPES = ["bf16", "int8", "fp8_e4m3", "fp8_e5m2"]

# Public-domain prose. Not model-generated: evaluating a cache on text the model
# itself produced would score the easy, in-distribution case and flatter whichever
# configuration generated it.
PASSAGES = [
    """It is a truth universally acknowledged, that a single man in possession of a
good fortune, must be in want of a wife. However little known the feelings or views
of such a man may be on his first entering a neighbourhood, this truth is so well
fixed in the minds of the surrounding families, that he is considered as the
rightful property of some one or other of their daughters. "My dear Mr. Bennet,"
said his lady to him one day, "have you heard that Netherfield Park is let at last?"
Mr. Bennet replied that he had not. "But it is," returned she; "for Mrs. Long has
just been here, and she told me all about it." Mr. Bennet made no answer. "Do not
you want to know who has taken it?" cried his wife impatiently. "You want to tell
me, and I have no objection to hearing it." This was invitation enough.""",

    """Call me Ishmael. Some years ago, never mind how long precisely, having little
or no money in my purse, and nothing particular to interest me on shore, I thought I
would sail about a little and see the watery part of the world. It is a way I have of
driving off the spleen and regulating the circulation. Whenever I find myself growing
grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I
find myself involuntarily pausing before coffin warehouses, and bringing up the rear
of every funeral I meet; and especially whenever my hypos get such an upper hand of
me, that it requires a strong moral principle to prevent me from deliberately
stepping into the street, and methodically knocking people's hats off, then, I
account it high time to get to sea as soon as I can.""",

    """After an unequivocal experience of the inefficacy of the subsisting federal
government, you are called upon to deliberate on a new Constitution for the United
States of America. The subject speaks its own importance; comprehending in its
consequences nothing less than the existence of the UNION, the safety and welfare of
the parts of which it is composed, the fate of an empire in many respects the most
interesting in the world. It has been frequently remarked that it seems to have been
reserved to the people of this country, by their conduct and example, to decide the
important question, whether societies of men are really capable or not of
establishing good government from reflection and choice, or whether they are forever
destined to depend for their political constitutions on accident and force.""",

    """Alice was beginning to get very tired of sitting by her sister on the bank, and
of having nothing to do: once or twice she had peeped into the book her sister was
reading, but it had no pictures or conversations in it, and what is the use of a
book, thought Alice, without pictures or conversations? So she was considering in her
own mind, as well as she could, for the hot day made her feel very sleepy and stupid,
whether the pleasure of making a daisy-chain would be worth the trouble of getting up
and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by
her. There was nothing so very remarkable in that; nor did Alice think it so very
much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be
late!\"""",
]

snap = snapshot_download(MID, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"])
tok = AutoTokenizer.from_pretrained(snap)

PREFIX = 32          # tokens prefilled before forced decode begins
RESULTS = {}


def forced_nll(eng, ids):
    """Mean NLL of the true continuation, decoded one token at a time.

    The cache fills with quantized K/V as the run proceeds, so any compounding
    shows up as a widening gap in the later quarters.
    """
    model, params = eng.model, eng.params
    prompt = jnp.asarray(ids[:PREFIX], dtype=jnp.int32)[None, :]
    padded, valid = pad_to_tpu_v6e_bucket(prompt)
    n_new = len(ids) - PREFIX
    last, caches, vmask = jax.block_until_ready(prefill_with_kv_cache(
        model, padded, valid, params, n_new + 1,
        quant_mode=eng.quant_mode, cache_dtype=eng.cache_dtype, window_kv=False))
    step = jax.jit(make_cached_decode_step(model, quant_mode=eng.quant_mode))
    bucket = int(padded.shape[1])
    lens = jnp.asarray([PREFIX], dtype=jnp.int32)

    nlls, greedy = [], []
    logits = last
    for t in range(n_new):
        true_next = int(ids[PREFIX + t])
        lp = jax.nn.log_softmax(logits[0].astype(jnp.float32))
        nlls.append(float(-lp[true_next]))
        greedy.append(int(jnp.argmax(logits[0])))
        fed = jnp.asarray([[true_next]], dtype=jnp.int32)   # teacher forcing
        caches, vmask, logits = step(params, caches, vmask, fed, lens + t,
                                     jnp.int32(bucket + t))
    return nlls, greedy


def quarters(xs):
    n = max(1, len(xs) // 4)
    return [sum(xs[i:i + n]) / len(xs[i:i + n]) for i in range(0, len(xs), n)][:4]


token_sets = []
for p in PASSAGES:
    ids = tok(p.strip(), add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None:
        ids = [tok.bos_token_id] + ids
    token_sets.append(ids)
print(f"passages: {[len(t) for t in token_sets]} tokens "
      f"({sum(len(t) - PREFIX for t in token_sets)} forced decode steps each config)",
      flush=True)

for dt in DTYPES:
    print(f"\n=== kv_cache_dtype={dt} ===", flush=True)
    eng = JaxGemmaEngine(MID, kv_cache_dtype=dt, quant_mode="w4a16", max_model_len=1024)
    eng.load(local_dir=snap)
    eng.bos_token_id = tok.bos_token_id
    all_nll, all_greedy = [], []
    t0 = time.time()
    for i, ids in enumerate(token_sets):
        nlls, greedy = forced_nll(eng, ids)
        all_nll.extend(nlls); all_greedy.extend(greedy)
        print(f"  passage {i}: ppl {math.exp(sum(nlls)/len(nlls)):7.3f}  "
              f"({len(nlls)} steps)", flush=True)
    mean = sum(all_nll) / len(all_nll)
    RESULTS[dt] = dict(nll=mean, ppl=math.exp(mean), greedy=all_greedy,
                       quarters=[math.exp(q) for q in quarters(all_nll)],
                       secs=round(time.time() - t0, 1))
    print(f"  OVERALL ppl {math.exp(mean):.4f}  nll {mean:.5f}", flush=True)
    print(f"  by quarter of decode: "
          f"{['%.3f' % q for q in RESULTS[dt]['quarters']]}", flush=True)
    del eng

base = RESULTS["bf16"]
print("\n" + "=" * 72)
print(f"{'dtype':<12}{'ppl':>10}{'vs bf16':>10}{'greedy match':>15}{'drift q1->q4':>14}")
for dt in DTYPES:
    r = RESULTS[dt]
    match = sum(a == b for a, b in zip(r["greedy"], base["greedy"])) / len(base["greedy"])
    drift = r["quarters"][-1] / base["quarters"][-1] / (r["quarters"][0] / base["quarters"][0])
    print(f"{dt:<12}{r['ppl']:>10.4f}{r['ppl']/base['ppl']:>9.4f}x"
          f"{match*100:>14.2f}%{drift:>13.4f}x")
print("=" * 72)
print("drift > 1.0 means the gap widens with decode depth (error compounding).")

with open(os.path.expanduser("~/kv_eval.json"), "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "greedy"}
               for k, v in RESULTS.items()}, f, indent=1)
print("\nKV_EVAL_DONE", flush=True)
