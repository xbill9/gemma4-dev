"""Does 8-bit KV error COMPOUND over a long decode?

The previous eval split its NLLs into quarters, but the four passages were
164/146/127/146 steps, so the quarter boundaries landed on passage boundaries
and each passage restarted the cache at 32 tokens. It measured passage
difficulty, not decode depth, and nothing accumulated past ~160 steps.

This runs ONE continuous forced decode of ~900 steps and bins by position within
that single run, so bin index is decode depth and the cache never resets. That is
the regime that matters for agent turns generating hundreds of tokens: a cache
whose error compounds looks perfect at token 50 and bad at token 800.

Reported per bin, against bf16 on the identical token stream:
  * NLL gap        — likelihood divergence at that depth
  * greedy match   — how often the argmax still agrees
Both are computed on the same forced sequence, so sampling divergence cannot
contaminate either.
"""
import os, sys, json, math, time
sys.path.insert(0, os.path.expanduser("~/gemma"))
import jax, jax.numpy as jnp
from jax_engine import JaxGemmaEngine
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ports.gemma4.jax_e_model import (make_cached_decode_step, prefill_with_kv_cache,
                                      pad_to_tpu_v6e_bucket)

MID = "google/gemma-4-E2B-it-qat-w4a16-ct"
DTYPES = ["bf16", "int8", "fp8_e4m3", "fp8_e5m2"]
PREFIX, N_BINS = 32, 6

# One continuous passage. Public domain (Melville, Austen, Carroll, Federalist),
# concatenated so the decode never restarts and depth is the only variable.
TEXT = """Call me Ishmael. Some years ago, never mind how long precisely, having
little or no money in my purse, and nothing particular to interest me on shore, I
thought I would sail about a little and see the watery part of the world. It is a way
I have of driving off the spleen and regulating the circulation. Whenever I find
myself growing grim about the mouth; whenever it is a damp, drizzly November in my
soul; whenever I find myself involuntarily pausing before coffin warehouses, and
bringing up the rear of every funeral I meet; and especially whenever my hypos get
such an upper hand of me, that it requires a strong moral principle to prevent me
from deliberately stepping into the street, and methodically knocking people's hats
off, then, I account it high time to get to sea as soon as I can. This is my
substitute for pistol and ball. With a philosophical flourish Cato throws himself
upon his sword; I quietly take to the ship. There is nothing surprising in this. If
they but knew it, almost all men in their degree, some time or other, cherish very
nearly the same feelings towards the ocean with me. There now is your insular city of
the Manhattoes, belted round by wharves as Indian isles by coral reefs; commerce
surrounds it with her surf. Right and left, the streets take you waterward. Its
extreme downtown is the battery, where that noble mole is washed by waves, and cooled
by breezes, which a few hours previous were out of sight of land. Look at the crowds
of water-gazers there. It is a truth universally acknowledged, that a single man in
possession of a good fortune, must be in want of a wife. However little known the
feelings or views of such a man may be on his first entering a neighbourhood, this
truth is so well fixed in the minds of the surrounding families, that he is
considered as the rightful property of some one or other of their daughters. My dear
Mr. Bennet, said his lady to him one day, have you heard that Netherfield Park is let
at last? Mr. Bennet replied that he had not. But it is, returned she; for Mrs. Long
has just been here, and she told me all about it. Mr. Bennet made no answer. Do not
you want to know who has taken it? cried his wife impatiently. You want to tell me,
and I have no objection to hearing it. This was invitation enough. Why, my dear, you
must know, Mrs. Long says that Netherfield is taken by a young man of large fortune
from the north of England; that he came down on Monday in a chaise and four to see
the place, and was so much delighted with it that he agreed with Mr. Morris
immediately; that he is to take possession before Michaelmas, and some of his
servants are to be in the house by the end of next week. Alice was beginning to get
very tired of sitting by her sister on the bank, and of having nothing to do: once or
twice she had peeped into the book her sister was reading, but it had no pictures or
conversations in it, and what is the use of a book, thought Alice, without pictures
or conversations? So she was considering in her own mind, as well as she could, for
the hot day made her feel very sleepy and stupid, whether the pleasure of making a
daisy-chain would be worth the trouble of getting up and picking the daisies, when
suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very
remarkable in that; nor did Alice think it so very much out of the way to hear the
Rabbit say to itself, Oh dear! Oh dear! I shall be late! But when the Rabbit actually
took a watch out of its waistcoat-pocket, and looked at it, and then hurried on,
Alice started to her feet, for it flashed across her mind that she had never before
seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and
burning with curiosity, she ran across the field after it, and fortunately was just
in time to see it pop down a large rabbit-hole under the hedge. After an unequivocal
experience of the inefficacy of the subsisting federal government, you are called
upon to deliberate on a new Constitution for the United States of America. The
subject speaks its own importance; comprehending in its consequences nothing less
than the existence of the UNION, the safety and welfare of the parts of which it is
composed, the fate of an empire in many respects the most interesting in the world.
It has been frequently remarked that it seems to have been reserved to the people of
this country, by their conduct and example, to decide the important question, whether
societies of men are really capable or not of establishing good government from
reflection and choice, or whether they are forever destined to depend for their
political constitutions on accident and force."""

snap = snapshot_download(MID, allow_patterns=["*.safetensors", "*.json", "*.model", "tokenizer*"])
tok = AutoTokenizer.from_pretrained(snap)
ids = tok(" ".join(TEXT.split()), add_special_tokens=False)["input_ids"]
if tok.bos_token_id is not None:
    ids = [tok.bos_token_id] + ids
N = len(ids) - PREFIX
print(f"single continuous run: {len(ids)} tokens, {N} forced decode steps, "
      f"{N // N_BINS} per bin", flush=True)

RESULTS = {}
for dt in DTYPES:
    print(f"\n=== {dt} ===", flush=True)
    eng = JaxGemmaEngine(MID, kv_cache_dtype=dt, quant_mode="w4a16",
                         max_model_len=len(ids) + 64)
    eng.load(local_dir=snap)
    prompt = jnp.asarray(ids[:PREFIX], dtype=jnp.int32)[None, :]
    padded, valid = pad_to_tpu_v6e_bucket(prompt)
    last, caches, vmask = jax.block_until_ready(prefill_with_kv_cache(
        eng.model, padded, valid, eng.params, N + 1,
        quant_mode=eng.quant_mode, cache_dtype=eng.cache_dtype, window_kv=False))
    step = jax.jit(make_cached_decode_step(eng.model, quant_mode=eng.quant_mode))
    bucket = int(padded.shape[1])
    lens = jnp.asarray([PREFIX], dtype=jnp.int32)

    nlls, greedy = [], []
    logits, t0 = last, time.time()
    for t in range(N):
        true_next = int(ids[PREFIX + t])
        lp = jax.nn.log_softmax(logits[0].astype(jnp.float32))
        nlls.append(float(-lp[true_next]))
        greedy.append(int(jnp.argmax(logits[0])))
        caches, vmask, logits = step(
            eng.params, caches, vmask,
            jnp.asarray([[true_next]], dtype=jnp.int32), lens + t,
            jnp.int32(bucket + t))
    RESULTS[dt] = dict(nlls=nlls, greedy=greedy, secs=round(time.time() - t0, 1))
    print(f"  ppl {math.exp(sum(nlls)/len(nlls)):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    del eng

base = RESULTS["bf16"]
per = N // N_BINS
print("\n" + "=" * 78)
print("NLL GAP vs bf16, by decode depth (one continuous run, cache never resets)")
print(f"{'depth':<16}" + "".join(f"{d:>15}" for d in DTYPES[1:]))
for b in range(N_BINS):
    lo, hi = b * per, (b + 1) * per if b < N_BINS - 1 else N
    row = f"{f'{lo}-{hi}':<16}"
    for dt in DTYPES[1:]:
        gap = (sum(RESULTS[dt]["nlls"][lo:hi]) - sum(base["nlls"][lo:hi])) / (hi - lo)
        row += f"{gap:>+15.5f}"
    print(row)

print("\nGREEDY MATCH vs bf16, by decode depth")
print(f"{'depth':<16}" + "".join(f"{d:>15}" for d in DTYPES[1:]))
for b in range(N_BINS):
    lo, hi = b * per, (b + 1) * per if b < N_BINS - 1 else N
    row = f"{f'{lo}-{hi}':<16}"
    for dt in DTYPES[1:]:
        m = sum(a == c for a, c in zip(RESULTS[dt]["greedy"][lo:hi], base["greedy"][lo:hi]))
        row += f"{m/(hi-lo)*100:>14.2f}%"
    print(row)
print("=" * 78)
print("A rising NLL gap or falling greedy match across rows = error compounding.")
print("Flat rows = the cache is stable at depth.")

with open(os.path.expanduser("~/kv_drift.json"), "w") as f:
    json.dump({k: {"secs": v["secs"], "nlls": v["nlls"], "greedy": v["greedy"]}
               for k, v in RESULTS.items()}, f)
print("\nKV_DRIFT_DONE", flush=True)
