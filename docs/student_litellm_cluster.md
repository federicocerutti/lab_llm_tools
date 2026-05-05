# Bia OpenAI-Compatible Student Guide

This guide is for the three classroom chat models exposed by the cluster:

- `qwen35-4b`
- `ministral3-3b`
- `phi4-mini`

These models are used through the cluster's OpenAI-compatible API at:

```bash
https://gpustack.ing.unibs.it/v1
```

## 1. Set your environment

Use your instructor-provided API key and the cluster base URL:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://gpustack.ing.unibs.it/v1"
```

The test script also accepts `LITELLM_API_KEY` for compatibility with older lab setups.

## 2. Make a simple chat request with `curl`

Choose one of the three model names and send a chat completion request:

```bash
curl -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "model": "qwen35-4b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Give me three study tips for prompt engineering."}
    ]
  }'
```

To use another model, replace `qwen35-4b` with `ministral3-3b` or `phi4-mini`.

## 3. Use Python

The cluster is OpenAI-compatible, so the standard OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="https://gpustack.ing.unibs.it/v1",
)

response = client.chat.completions.create(
    model="phi4-mini",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain what a language model is in one sentence."},
    ],
)

print(response.choices[0].message.content)
```

Change the `model` field to `qwen35-4b`, `ministral3-3b`, or `phi4-mini` as needed.

## 4. Test all three models at once

Run the small LiteLLM checker script from this repository:

```bash
python scripts/test_cluster_models.py
```

The script sends 10 concurrent requests to each classroom model, for 30 requests total, and prints:

- whether each request succeeded
- how long each response took
- the returned text
- per-model latency summaries

If a model returns an empty answer, the checker counts that request as a failure.

If you get an authentication error, check that your API key is valid and that `OPENAI_BASE_URL` is set to `https://gpustack.ing.unibs.it/v1`.

## 5. Practical notes

- These are chat models, so use the `/chat/completions` endpoint.
- Pick one model per request.
- Start with short prompts while you are testing your setup.
- If a request fails, the first things to verify are the API key, the base URL, and the exact model name.
