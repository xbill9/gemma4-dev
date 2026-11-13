Title: Can't Beat Free: Gemma 4 31B on AMD Token Lab, with a local chat page and an ADK agent
Category: Artificial Intelligence (AI) > AI Developer Program
Tags: ai, developer-program, llm

---

Hi all, I'm xbill, a Google Developer Expert. I've been building on **AMD Token Lab**, the free API that comes with the AMD AI Developer Program, and wrote up everything from sign-up to a working agent.

![Can't Beat Free](https://raw.githubusercontent.com/xbill9/gemma4-dev/main/docs/tokenlab-free/devto-cover.aff96a99.jpg)

**Why Token Lab**
Joining the program is free, and Token Lab needs no credit request. A key comes with 10 points, a 30-day access period and 20 requests per minute, and the catalog had 10 models, all marked Free. I used Gemma 4 31B: text and images, thinking, tool calling and a 262,144-token context on an OpenAI-compatible API (`https://api.tokenlab.amd.com/v1`). Everything in the article used 0.000335 of the 10 points over 23 requests.

**What I built on it**
• **A local chat page**: standard-library Python that serves on 127.0.0.1 and holds the key in a small proxy, so it never reaches the browser. It shows each reply's first-token time, decode rate, tokens and points, and takes images.
• **A terminal client**: one question, a pipe (`git log | ask-tokenlab summarize this`), or a conversation with memory.
• **An ADK agent**: Google's Agent Development Kit reaches Token Lab through LiteLLM with one `LiteLlm(model="openai/Gemma-4-31B-IT", api_base=...)` object. It has two tools, and Gemma called them correctly, including answering a counting question from the tool's exact result.

**Good to know**
• Gemma 4 31B is served by SGLang behind Token Lab, per its model card.
• The endpoint is shared: one request decoded at 2.0 to 9.5 tokens per second on 2026-11-13, and a request over the model's 128-at-once limit gets a 429 to retry.
• Regenerating a key keeps your points and access period, and the access period can be extended once by 15 days.

**How to sign up**
1. Join the AMD AI Developer Program at https://developer.amd.com (sign-up form, then verify your email).
2. Sign in at https://notebooks.amd.com/tokenlab with the AI Developer Program login.
3. Copy your key and the base URL, and call a model from the catalog.

**Full write-up**
dev.to: https://dev.to/gde/cant-beat-free-gemma-4-31b-on-amd-token-lab-with-a-local-chat-page-and-an-adk-agent-c5b
Medium: https://medium.com/@xbill999/cant-beat-free-gemma-4-31b-on-amd-token-lab-with-a-local-chat-page-and-an-adk-agent-0ae3911bf0d7
Code: https://github.com/xbill9/gemma4-dev/tree/main/tokenlab-sglang-amd-31b

Thanks to the AMD team for making this free. Happy to answer questions!
