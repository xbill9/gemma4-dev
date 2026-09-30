# Demo: one interactive user, Gemma 4 E2B on a GTX 1650 Ti

About 8 minutes. One user, one conversation, served locally from a 4 GB laptop GPU.
Uses `~/bin/gpu` (start/stop/check the server) and `~/bin/ask` (talk to it). Every
serving value comes from this rig's `tpu.env`: the v2 exact GGUF, `-ngl 99 -fa 1`,
`--parallel 1`, thinking off.

## Before you start (5 minutes ahead)

```
gpu start          # starts the server and checks it's really the GPU one
ask -q "hi"        # warm-up, so the model file is already in memory
```

Keep a second terminal running `watch -n1 nvidia-smi` for step 3.

## 1. It's instant (30 s)

```
ask What is the capital of Australia? One sentence.
```

The whole answer is back in about 0.4 s (rehearsed 2026-09-29: 0.39 s end to end; 2026-09-30: 0.32 s); longer ones
stream at ~75–80 tok/s. Point out that it runs on a 4 GB laptop GPU with no cloud involved.

## 2. Pipe it real work (1 min)

```
git -C ~/gemma4-dev log --oneline -12 | ask summarize what this project has been working on, in 3 bullets
git -C ~/gemma4-dev show --stat --format='%s%n%n%b' HEAD | ask write a one-line summary of this commit
```

About 4 s each in rehearsal (2026-09-30: 4.3 s and 4.5 s; 2026-09-29: 5 s and 4 s). Keep the input small: the context is 8192 tokens, and prompt
processing runs at ~345 tok/s, so a 4,000-token paste is ~12 s of silence before the first word.
Do not pipe `git diff` of a benchmark commit; it can overflow the context outright.

## 3. The footprint (30 s)

Switch to the `nvidia-smi` terminal: about 1.5 GB of 4 GB in use (1491 MiB in rehearsal, 1501 MiB on 2026-09-30), and
30–40 W while it answers. Then:

```
gpu arm
```

It reports, from the live process, which device is actually serving. The CPU twin rig uses the
same port, so this is checked rather than assumed.

## 4. A conversation (2 min)

Run `ask` with no arguments to open an interactive session with memory:

- "Explain what quantizing a model means, in three sentences, for an engineer."
- "Give me an everyday analogy for it."
- "Now as a haiku."

Follow-ups are fast because llama-server reuses the prompt cache: only the new text gets processed.
`!` clears the conversation; Ctrl-D leaves.

Avoid asking it about itself (its training data, how it was built): in rehearsal it confidently
invented a "proprietary fine-tuning dataset". Keep questions on general topics.

## 5. Turn thinking on (1 min)

```
ask -T Which is larger, 9.11 or 9.9? Explain.
```

The dimmed reasoning streams first, then the answer. It's off by default because it delays the
answer: in rehearsal this question spent 700–1,100 characters thinking before the first word, and
8.8 s end to end on 2026-09-30. The
reasoning only shows when `ask` is writing to a terminal.

## 6. The story behind the file (1 min, optional)

- Google's own Q4_0 GGUF moves the QAT-trained weights off their grid, and stores the embeddings
  at Q6_K. This file is rebuilt exactly from Google's QAT checkpoint instead.
- **About 30x closer to the original model:** mean KL divergence from bf16 is 0.054 for Google's
  file and 0.0017 for this one (CUDA, wikitext-2).
- **11–12% faster decode** on this card and **22% smaller** (3.35 → 2.62 GB).
- Published as `xbill9/gemma-4-E2B-it-qat-q4_0-exact-gguf`; anyone can rebuild it byte for byte.
  Evidence in `benchmarks/runs/2026-09-29-exact-gguf-v2-1650ti/`.

## Safety net

- Don't use `-n` with small values; answers are capped at 1024 tokens by default.
- If anything hangs: `gpu stop && gpu start` takes a few seconds.
- The CPU twin rig uses the same port (8080). Don't have both up; `gpu start` refuses if the port
  is taken.
- Avoid questions that need current events. It's a small model with a fixed knowledge cutoff, and
  it will answer confidently anyway.
