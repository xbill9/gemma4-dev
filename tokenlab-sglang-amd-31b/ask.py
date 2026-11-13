#!/usr/bin/env python3
"""Ask Gemma 4 31B on AMD Token Lab from a terminal. Standard library only.

    ask-tokenlab [-s system] [-n maxtokens] [-t temp] [-T] [-q] [-r] [-i] [-I image] [prompt ...]
    <something> | ask-tokenlab [prompt ...]

The terminal sister of ask-1650ti / ask-t4. With no prompt and a terminal on stdin it is a
REPL that keeps the turns in context ('!' forgets them, Ctrl-D or 'quit' leaves). With a
prompt it answers once and exits, so it works in a pipe: piped stdin becomes the material
and the arguments the instruction about it.

  git diff | ask-tokenlab write a commit message

Thinking is off unless -T (chat_template_kwargs enable_thinking=true). With it on, the
reasoning streams dimmed on stderr when stderr is a terminal (-q hides it) and only the
answer goes to stdout and into the REPL history.

429 (Token Lab's per-model concurrency limit or the key's rate limit) and 502/503/504 are
retried before the reply starts, honouring Retry-After; with stderr on a terminal each
retry is noted. A stream that ends without [DONE] was cut off and is reported as such.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import tokenlab  # noqa: E402

DIM, RESET = "\033[2m", "\033[0m"


def image_part(path):
    import base64
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def stream_reply(cfg, args, messages):
    """Stream one reply; returns the answer text ('' on failure)."""
    body = {
        "model": cfg["MODEL_NAME"],
        "messages": messages,
        "max_tokens": args.n,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": args.T},
    }
    if args.t is not None:
        body["temperature"] = args.t
    tty_err = sys.stderr.isatty()

    def on_retry(attempt, code, delay):
        if tty_err:
            print(f"{DIM}[{code}, retry {attempt} in {delay:g}s]{RESET}", file=sys.stderr, flush=True)

    if args.r:
        body["stream"] = False
        body.pop("stream_options")
    try:
        resp = tokenlab.open_api(cfg, "/chat/completions", json.dumps(body).encode(), on_retry=on_retry)
    except (OSError, RuntimeError) as e:
        print(f"ask-tokenlab: {e}", file=sys.stderr)
        return ""
    if resp.status != 200:
        print(f"ask-tokenlab: HTTP {resp.status}: {tokenlab.error_message(resp)}", file=sys.stderr)
        return ""
    if args.r:
        j = json.loads(resp.read())
        print(json.dumps(j, indent=2))
        return (j.get("choices") or [{}])[0].get("message", {}).get("content") or ""

    answer, thinking, done, finish = [], False, False, None
    show_think = args.T and tty_err and not args.q
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue  # ': ping' keep-alives, 'id:' and 'event:' lines
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            j = json.loads(data)
            if j.get("error"):
                print(f"\nask-tokenlab: {j['error'].get('message', j['error'])}", file=sys.stderr)
                break
            for ch in j.get("choices", []):
                d = ch.get("delta") or {}
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc and show_think:
                    if not thinking:
                        sys.stderr.write(DIM)
                        thinking = True
                    sys.stderr.write(rc)
                    sys.stderr.flush()
                if d.get("content"):
                    if thinking:
                        sys.stderr.write(RESET + "\n")
                        thinking = False
                    answer.append(d["content"])
                    sys.stdout.write(d["content"])
                    sys.stdout.flush()
                finish = ch.get("finish_reason") or finish
    if thinking:
        sys.stderr.write(RESET + "\n")
    sys.stdout.write("\n")
    if not done:
        print("ask-tokenlab: the stream ended without [DONE]; send it again", file=sys.stderr)
    elif not answer and finish == "length":
        print(f"ask-tokenlab: max_tokens ({args.n}) ran out before the answer began; raise -n", file=sys.stderr)
    return "".join(answer)


def repl(cfg, args, base):
    try:
        import readline  # noqa: F401  (line editing for input())
    except ImportError:
        pass
    messages = list(base)
    print(f"ask-tokenlab: {cfg['MODEL_NAME']} at {cfg['TOKENLAB_BASE_URL']}")
    print("ask-tokenlab: Ctrl-D or 'quit' to leave, '!' to forget the conversation so far")
    while True:
        try:
            line = input("> ")
        except EOFError:
            print()
            return
        if not line.strip():
            continue
        if line.strip() in ("quit", "exit"):
            return
        if line.strip() == "!":
            messages = list(base)
            print("ask-tokenlab: context cleared")
            continue
        messages.append({"role": "user", "content": line})
        reply = stream_reply(cfg, args, messages)
        if reply:
            messages.append({"role": "assistant", "content": reply})
        else:
            messages.pop()


def main():
    p = argparse.ArgumentParser(prog="ask-tokenlab", description="Ask Gemma 4 31B on AMD Token Lab.")
    p.add_argument("-s", metavar="system", default=os.environ.get("ASK_TOKENLAB_SYSTEM"))
    p.add_argument("-n", metavar="maxtokens", type=int, default=int(os.environ.get("ASK_TOKENLAB_MAXTOKENS", 2048)))
    p.add_argument("-t", metavar="temp", type=float, default=None)
    p.add_argument("-T", action="store_true", help="thinking on for this request")
    p.add_argument("-q", action="store_true", help="hide the reasoning stream")
    p.add_argument("-r", action="store_true", help="print the raw JSON reply (no stream)")
    p.add_argument("-i", action="store_true", help="interactive even with a prompt-less pipe")
    p.add_argument("-I", metavar="image", help="attach an image file to the prompt")
    p.add_argument("prompt", nargs="*")
    args = p.parse_args()
    cfg = tokenlab.load_config()
    base = [{"role": "system", "content": args.s}] if args.s else []

    text = " ".join(args.prompt)
    if args.i or (not text and sys.stdin.isatty()):
        repl(cfg, args, base)
        return 0
    stdin = "" if sys.stdin.isatty() else sys.stdin.read()
    prompt = f"{text}\n\n{stdin}" if text and stdin else text + stdin
    content = [image_part(args.I), {"type": "text", "text": prompt}] if args.I else prompt
    return 0 if stream_reply(cfg, args, base + [{"role": "user", "content": content}]) else 1


if __name__ == "__main__":
    sys.exit(main())
