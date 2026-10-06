curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-12B-it",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 15,
    "logprobs": true,
    "top_logprobs": 2
  }'
