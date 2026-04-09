# Token Streaming (SSE)

LLMesh supports real-time token streaming via Server-Sent Events (SSE). Tokens arrive at the client as they are generated, rather than waiting for the full response.

---

## Usage

Set `stream: true` in your request to `POST /v1/chat/completions`:

```bash
curl -N -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model": "llama3:8b", "messages": [{"role": "user", "content": "Hello"}], "stream": true}' \
     https://your-hub/v1/chat/completions
```

The response is an SSE stream in OpenAI-compatible format:

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":8}}

data: [DONE]
```

Non-streaming requests (`stream: false` or omitted) are unaffected.

---

## Backend Support

| Backend | Streaming | Notes |
|---|---|---|
| Ollama | Supported | Full token-by-token streaming |
| vLLM | Beta | Falls back to blocking — result delivered as a single SSE frame |
| MLX | Beta | Falls back to blocking — result delivered as a single SSE frame |

---

## Nginx Configuration

**`proxy_buffering off` is required for SSE endpoints.** Without it, nginx buffers the entire response until inference completes, defeating streaming.

See [nginx_deployment.md](nginx_deployment.md) for the full location block configuration. The streaming location block must appear before the general `location /` block.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `STREAM_CHUNK_TIMEOUT` | `300` | Seconds the hub waits per token before declaring the node unresponsive |
| `TASK_TTL_SECONDS` | `3600` | Seconds before completed/failed tasks are purged from memory |

---

## Limitations

- **Ollama only for now.** vLLM and MLX streaming is deferred — these backends deliver a single result frame instead of token-by-token output.
- **Anthropic endpoint does not stream.** `POST /v1/messages` returns HTTP 400 if `stream: true` is passed.
- **No mid-stream retry.** If the node fails during streaming, the client must retry the full request.
- **In-memory stream state.** Hub restart drops all active streams.
