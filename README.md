# fma02 free-model aggregator

One Node 22 server that stands in for a paid LLM API and answers every completion from
OpenRouter's free models, dialled through the grader's gateway. Built for make.fun's
`free-model-aggregator-02` bounty.

## What it does

- `POST /v1/chat/completions` — OpenAI Chat Completions wire, JSON and SSE streaming
- `POST /v1/messages` — Anthropic Messages wire, JSON and SSE streaming; the body is
  translated to Chat Completions for the gateway and the answer is translated back
- `GET /v1/models` — the model list in the wire the auth header names
  (`Authorization: Bearer` → OpenAI shape, `x-api-key` → Anthropic shape)
- `node main.ts --list` — prints the catalog document byte for byte as it registers it
- `GET $MAKEFUN_ENV_URL/work` is polled; the process exits 0 when the env says done

## Routing

Candidates are the brief's free models ordered by the house's `rating_milli`, best first.
A request's requirements are read off its body — `response_format` (JSON mode), `tools`,
image blocks — and models that meet them rank ahead of models that do not. The
best-rated candidate that can serve the prompt is dialled first; every refusal
(HTTP 4xx/5xx, or an answer with no content) falls through to the next candidate, and a
5xx is retried once on the same model before failing over. A recovered refusal is
reported in `x-makefun-upstream-failures`, the header the failover column reads.

## Determinism

Every choice is a pure function of the brief, the requests and the gateway's answers:
no wall clock, no unseeded randomness. A replayed trial reproduces the live run's answers
and state export byte for byte, as the final's replay pass requires.

## Layout

- `main.ts` — server, env registration, prompt polling, routing and failover loop
- `lib/wire.ts` — both client wires: shapes, streaming event order, translation between
  Anthropic Messages and Chat Completions, usage and stop reasons kept whole
- `lib/providers.ts` — the brief reader, capability checks, the catalog document and the
  model ordering

## Run

```
MAKEFUN_PROMPT_FILE=brief.json MAKEFUN_ENV_URL=http://… node main.ts
```

With neither variable set, `node main.ts --list` prints an empty catalog and exits 0.

## License

MIT
