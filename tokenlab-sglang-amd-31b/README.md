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

#### A sample ADK agent

`tokenlab_agent/` is a small [ADK](https://google.github.io/adk-docs/) agent whose model is Gemma 4 31B
on Token Lab, reached through LiteLLM's OpenAI provider:

```python
LiteLlm(model="openai/Gemma-4-31B-IT", api_base="https://api.tokenlab.amd.com/v1",
        api_key=TOKENLAB_API_KEY, num_retries=30,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

It has two tools: `list_tokenlab_models` (what Token Lab serves now) and `count_and_range`, which does
counting and filtering in code so the model quotes an exact result.

```bash
pip install -r requirements.txt     # google-adk, litellm
make agent                          # adk run, in the terminal
make agent-web                      # adk web on http://localhost:8767
make agent-once Q='"How many of 0, 2, 3, 20, 21, 22, 23, 10, 11, 12, 13 are 10 or more?"'
```

On 2026-11-13 that question took 14.4 s: Gemma called `count_and_range` with `at_least=10`, the tool
returned 8, and the answer quoted 8 with the filter `10 <= x`.
