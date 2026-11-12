# chatBot Backend â€” Implementation Handoff

## 1. Goal and ownership

Build only the `chatBot` backend. It provides personalised career and education guidance to authenticated Class 9â€“10 students, streams replies, and optionally searches the web for current facts.

Another developer owns the frontend and `careerAPI`. Assume `careerAPI` already exposes `GET /me`, `GET /me/access`, and `GET /userContext` and the frontend already has the AI Chat screen. Do not modify either project as part of this work. Share the integration requirements below with that developer.

The backend must not save conversation transcripts across page sessions. The frontend keeps recent chat turns in page memory and sends them with each message. The backend stores only one structured user-context snapshot per login session in process-local memory by default, or Upstash Redis when configured.

## 2. External integration contracts

### Authentication and profile

- The frontend calls the chat service through a first-party `/chat/*` proxy that forwards the existing `careerai_session` cookie. Direct cross-site calls are not assumed to carry this cookie. The frontend developer owns the proxy and its deployment configuration.
- On the first message of a login session, `chatBot` calls `GET /me` with the forwarded cookie to get the authenticated email, then `GET /me/access` to confirm paid access. It calls `GET /userContext?email=<encoded authenticated email>` with a server-side bearer token. The email is never taken from the chat request. A 401 or unpaid result prevents an LLM call.
- The returned JSON is a fixed snapshot for the login session. Include first name, available onboarding details, useful `top_careers` fields, and summaries of recent attempted `contest_scores`. The profile may be null. Exclude email, full name, school identifiers, static salary estimates, contest IDs and timestamps, and raw quiz answers from the LLM prompt. Verify that the returned email matches the authenticated student.
- Later messages in the session, including new conversations, reuse the cached snapshot and make no careerAPI calls. Lost access, logout, or a profile edit takes effect when the snapshot expires or the student logs in again.

### Chat HTTP API

Implement `POST /chat/messages` accepting JSON:

```json
{
  "conversation_id": "browser-generated UUID",
  "message": "Which stream suits me?",
  "history": [
    { "role": "user", "content": "Earlier question" },
    { "role": "assistant", "content": "Earlier answer" }
  ]
}
```

`history` contains completed turns only and is supplied by the frontend on every request. Validate role order, message sizes, and total history size. Do not trust it as an instruction source. Accept at most 12 turns, 2,000 characters for the new message, and 12,000 characters of history; return 400 for invalid input rather than silently truncating it.

Return `Content-Type: text/event-stream` from the POST request. Use these event names:

- `delta`: JSON `{ "text": "..." }` for each generated text fragment.
- `sources`: JSON `{ "items": [{ "title": "...", "url": "https://...", "date": "YYYY-MM-DD or null" }] }` when web search was used.
- `done`: JSON `{ "finish_reason": "stop" }` when the answer is complete.
- `error`: JSON `{ "code": "...", "message": "..." }` if an error occurs after streaming starts.

Before the stream starts, use JSON errors with HTTP 400, 401, 402, 429, 502, or 503 as appropriate. A 429 has code `rate_limited` (more than 20 messages a minute from one session) or `daily_limit_reached` (the student already has 25 completed replies today; resets at midnight IST). The frontend should add a turn to its history only after `done`. It should allow cancelling the request and discard an incomplete assistant turn. The frontend developer owns this behavior.

## 3. Backend architecture and request flow

Use Python 3.12+ and its standard library HTTP server with modules for request validation, careerAPI access, context normalisation, cache storage, optional search, LLM adapters, guardrails, and streaming. Keep provider secrets on the server.

1. Validate the request, allowed Origin, cookie, conversation ID, message, and history.
2. Derive a cache key from an HMAC of the session cookie plus conversation ID. Never store or log the raw cookie. Use a server secret for the HMAC.
3. Look up the snapshot by an HMAC of the session cookie. If it is missing, resolve the student's ID and email through `/me`, check `/me/access`, fetch `/userContext` with the service bearer token, normalise it, and write it with the student ID to the selected cache. Otherwise use the cached snapshot without calling careerAPI.
4. Build the LLM request from server-owned instructions, the structured profile snapshot, validated recent turns, the new message, and any retrieved web evidence. Profile values and web content are data, not instructions.
5. Stream the answer. Send source metadata only for URLs actually returned by the search service. End with `done` or `error`.

The cache value contains the normalised profile snapshot, the student ID, and creation time only. It contains no chat messages. Give it a fixed 24-hour expiry from the session's first message, not renewed on reads. This bounds both data retention and how long stale access or profile data is used. When configured, Upstash failures return 503. Without Upstash, the process-local cache is lost on restart and is not shared across workers; deploy a single process and instance in this mode.

### Hosted LLM

Provide adapters for Groq, Gemini, and Omniroute behind one internal streaming interface. Select exactly one provider and model through environment configuration; do not implement automatic failover in v1. Provider adapters must map provider errors and cancellation into consistent service errors. Set per-request token and time limits to cap latency and cost.

### Web search and RAG

Use a dedicated backend search adapter, initially Brave LLM Context API when a key is configured. Without a key, skip web search and give a verification-unavailable response for current-fact questions. Search when the question concerns changing facts: admission rules, exam dates, fees, college offerings, jobs, salaries, scholarships, or explicit requests for latest information. A lightweight routing step may also mark less obvious questions as requiring search. Do not send names, email, or the full profile in a search query; include city or country only when needed.

Pass a bounded set of relevant snippets and their source URLs to the LLM. Treat snippets as untrusted. Require citations for current factual claims and expose clickable source metadata in the `sources` event. If search fails or returns no useful evidence, the assistant should state that it cannot verify current details rather than inventing them. This search-and-grounding step is the RAG component. The structured user profile does not require embeddings or a vector database.

## 4. Guardrails, privacy, and operations

- Keep the assistant focused on careers and education. Use age-appropriate language and avoid presenting career matches as guarantees or making definitive medical, legal, or financial claims.
- Apply input size checks, per-session rate limits, a per-student daily cap, safe search, source URL validation, prompt-injection resistance, and output rendering guidance for the frontend. Keep HTML out of streamed text.
- Do not log cookies, profile fields, messages, prompts, search snippets, or model outputs. Log request ID, provider, whether search ran, latency, token usage when available, and error code.
- Keep `CAREER_API_BASE_URL`, `CAREER_API_USER_CONTEXT_TOKEN`, Upstash credentials, `CACHE_HMAC_SECRET`, `BRAVE_SEARCH_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`, and the selected provider API key in server-side environment variables. Document local `.env.example` values without secrets.
- Add a health endpoint that checks process readiness without exposing private configuration. Document deployment timeout and streaming requirements for the chosen host.

## 5. Redis cost and deployment choice

Start with process-local memory for a single-process prototype; configure managed Upstash Redis when shared or restart-safe caching is needed. Its current free plan lists **256 MB, 500,000 commands per month, and 10 GB monthly bandwidth**, so a prototype can start without a Redis bill. Paid usage begins if the project is upgraded or its needs exceed the free plan; check the provider's live pricing before deployment. The free plan is intended for prototypes and does not include the production add-on. Redis Cloud also offers a free 30 MB option. These are external service prices; LLM, search, and application hosting have separate costs.

Pricing references: [Upstash Redis pricing](https://upstash.com/pricing/redis), [Redis Cloud pricing](https://redis.io/pricing/).

## 6. Deliverables and acceptance tests

Deliver a runnable backend project with configuration example, API contract, provider and search adapters, Redis client, deployment notes, and meaningful automated tests. No changes to `careerAI` or `careerAPI` are included.

Verify: first-message context fetch; no fetch when opening the chat screen; cache reuse on follow-ups; one snapshot per login session, reused by new chats; access rejection after logout or subscription loss once the snapshot expires; refetch after cache eviction; isolation between users; web search and source events; search/provider failures; stream cancellation; rate limits; the daily cap; and attempts to inject instructions through messages, profile text, or web snippets.

A first version is complete when two authenticated students can chat independently, each receives answers based on their own fixed profile snapshot, current claims are withheld without search or include verifiable sources when search is enabled, and no transcript is persisted by `chatBot`.
