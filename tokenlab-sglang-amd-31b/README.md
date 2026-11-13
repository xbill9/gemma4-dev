# tokenlab-sglang-amd-31b

Gemma 4 31B on AMD GPUs through **AMD Token Lab**, a hosted OpenAI-compatible API, with the
same local chat page as the v5e-1 vLLM rig and an `ask`-style terminal client like the 1650 Ti
rigs. Nothing to provision: get a key, put it in `.env`, start the page.

#### Start it up

**1. Get a key** at `notebooks.amd.com/tokenlab` (AMD AI Developer Program sign-in). It comes with
a 30-day access period and a points budget.

**2. Put it in `.env`** (gitignored, mode 0600):

```bash
umask 077; echo 'TOKENLAB_API_KEY=tl-...' > .env
```

**3. Open the chat page** at `http://localhost:8766`:

```bash
make ui
```

**4. Or ask from a terminal:**

```bash
make bin                                   # ~/bin/ask-tokenlab -> ask.py
ask-tokenlab What is the capital of Australia? One sentence.
git log --oneline -12 | ask-tokenlab summarize this in 3 bullets
ask-tokenlab -T Which is larger, 9.11 or 9.9?
ask-tokenlab -I photo.jpg what is in this picture?
ask-tokenlab                               # REPL with memory; '!' clears, Ctrl-D leaves
```

#### What is behind the endpoint

From `GET /v1/models` on 2026-11-13: `Gemma-4-31B-IT`, a "Dynamic sglang-router service", context
262,144 tokens, text and image input, parameters `temperature`, `max_tokens`, `top_p`, `stream`,
`response_format`, `tools`, `tool_choice`, `reasoning_effort`. Token Lab names the hardware only as
"AMD Instinct and Radeon".

#### The chat page

`chat_ui.py` is standard-library Python. It serves the page on 127.0.0.1 and forwards `/v1/*` to
Token Lab with the key attached, so the key never reaches the browser. It retries 429 and 502/503/504
before the reply starts, as Token Lab's FAQ asks. The side panel has max tokens, temperature, top-p,
thinking and a system prompt; the last reply's first-token time, decode rate, tokens and points; the
model card from `/v1/models`; and this session's totals. Attach an image with 🖼. A stream that ends
without `[DONE]` is flagged so you can send it again.
