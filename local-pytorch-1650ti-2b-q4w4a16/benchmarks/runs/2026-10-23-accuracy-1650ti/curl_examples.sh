#!/bin/bash
# Two requests against openai_server.py, as shown in the article: plain chat, then a tool call.
set -e
URL=http://127.0.0.1:8100/v1/chat/completions
curl -s $URL -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "In one sentence, what is a GPU?"}],
  "max_tokens": 64
}'
echo
curl -s $URL -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
  }}]
}'
echo
