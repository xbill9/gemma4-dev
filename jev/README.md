# jev — does DiffusionGemma's one-step read beat plain Gemma 4 at Jev-style decisions?

Background, sources and what is measured so far: [NOTES.md](NOTES.md).

The claim under test comes from Karl Weinmeister's *How to build a Jev-style classifier with DiffusionGemma and vLLM* (Medium / LinkedIn, 2026-09-22), built on vLLM PR #57250. The article shows speed on 8 alerts. It reports no accuracy and no calibration measurement, and it has no baseline.

Two readouts, one prompt:

- **diffusion** — `google/diffusiongemma-26B-A4B-it`, one denoise step over a canvas seeded with the answer template, read through the PR's own `structured_server.py` code (vendored in `vendor/` at merge commit `1b3b88ec`), with 4 noise draws per example.
- **autoregressive** — `google/gemma-4-26B-A4B-it`, the same token prefix, next-token logprobs at the answer slot.

Both use the PR's system prompt, answer template, label tokens and `slot_distribution`. `run_eval.py` checks that Gemma 4's chat template renders the identical prefix before it sends anything.

Data: 300 examples each from sst2, ag_news, emotion and tweet_eval irony, stratified with a fixed seed (`build_eval_set.py`, output in `data/`).

```bash
python3 build_eval_set.py                     # already run; data/ is committed
python3 run_eval.py --arm diffusion      --upstream http://HOST:8000 --model google/diffusiongemma-26B-A4B-it --run RUN
python3 run_eval.py --arm autoregressive --upstream http://HOST:8000 --model google/gemma-4-26B-A4B-it --run RUN
python3 score.py --run RUN                     # writes results/RUN/SUMMARY.md
python3 -m unittest discover -s tests -v
```

Serving both models needs a vLLM build that includes PR #57250 (merged 2026-09-22) and one GPU with at least 80 GB for bf16.
