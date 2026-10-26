# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM 0.28 (vllm/parser/gemma4.py and vllm/tool_parsers/gemma4_utils.py),
# Copyright contributors to the vLLM project; Copyright 2025 Google Inc. HuggingFace Inc. team.
"""Parse Gemma 4 tool calls out of decoded text into OpenAI `tool_calls`.

Gemma 4 writes a call as ``<|tool_call>call:name{key:value,...}<tool_call|>`` with strings fenced by ``<|"|>``.
vLLM keeps bare values (numbers, booleans) as strings; here a bare value is read as JSON when it parses, so
``count:42`` becomes 42 and ``flag:true`` becomes True.
"""

import json
import re
import uuid

STRING_DELIM = '<|"|>'
_DL = len(STRING_DELIM)
_CALL = re.compile(r"<\|tool_call\>call:([\w.\-]+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)", re.DOTALL)


def _bare(raw):
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _skip_string(s, i, n):
    nd = s.find(STRING_DELIM, i + _DL)
    return n if nd == -1 else nd + _DL


def _scan_block(s, i, n, open_ch, close_ch):
    """From s[i] == open_ch, return the index just past the matching close_ch (strings skipped)."""
    depth, i = 1, i + 1
    while i < n and depth > 0:
        if s[i : i + _DL] == STRING_DELIM:
            i = _skip_string(s, i, n)
            continue
        if s[i] == open_ch:
            depth += 1
        elif s[i] == close_ch:
            depth -= 1
        i += 1
    return i


def _value(s, i, n, stops):
    """Parse one value at s[i]; return (value, next index)."""
    if s[i : i + _DL] == STRING_DELIM:
        end = s.find(STRING_DELIM, i + _DL)
        end = n if end == -1 else end
        return s[i + _DL : end], min(n, end + _DL)
    if s[i] == "{":
        j = _scan_block(s, i, n, "{", "}")
        return parse_args(s[i + 1 : j - 1]), j
    if s[i] == "[":
        j = _scan_block(s, i, n, "[", "]")
        return parse_array(s[i + 1 : j - 1]), j
    j = i
    while j < n and s[j] not in stops:
        j += 1
    return _bare(s[i:j].strip()), j


def parse_args(s):
    """``key:value,...`` (Gemma 4's argument format) -> dict."""
    out, i, n = {}, 0, len(s)
    while i < n:
        while i < n and s[i] in " ,\n\t":
            i += 1
        if i >= n:
            break
        k = s.find(":", i)
        if k == -1:
            break
        key = s[i:k].strip()
        if key.startswith(STRING_DELIM) and key.endswith(STRING_DELIM):
            key = key[_DL:-_DL]
        i = k + 1
        while i < n and s[i] in " \n\t":
            i += 1
        if i >= n:
            out[key] = ""
            break
        out[key], j = _value(s, i, n, ",}]")
        if j <= i:
            break
        i = j
    return out


def parse_array(s):
    items, i, n = [], 0, len(s)
    while i < n:
        while i < n and s[i] in " ,\n\t":
            i += 1
        if i >= n:
            break
        v, j = _value(s, i, n, ",]")
        if j <= i:
            break
        items.append(v)
        i = j
    return items


def parse_tool_calls(text):
    """Decoded text (special tokens kept) -> (content without the calls, OpenAI tool_calls list)."""
    calls = [
        {
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": m.group(1), "arguments": json.dumps(parse_args(m.group(2)), ensure_ascii=False)},
        }
        for m in _CALL.finditer(text)
    ]
    content = _CALL.sub("", text) if calls else text
    return content, calls
