# Career chatbot backend

Python 3.12 service for personalised career and education chat. It streams Server-Sent Events (SSE) from one configured hosted LLM, reads the student's profile from `careerAPI` on the first message, and keeps that fixed profile in a process-local memory cache by default, or Upstash Redis when configured. The browser supplies completed chat turns; this service stores no transcript.

## Run locally

1. Copy `.env.example` to `.env` and fill in the required values. Leave both Upstash settings empty to use in-memory caching, and leave `BRAVE_SEARCH_API_KEY` empty to skip web search. Set `CACHE_HMAC_SECRET` to a random secret of at least 32 characters. Set `CAREER_API_USER_CONTEXT_TOKEN` to the server-side bearer token for `/userContext`. Set `LLM_PROVIDER` to `groq`, `gemini`, or `omniroute`, along with a model name and that provider's API key. For OmniRoute, set `OMNIROUTE_BASE_URL` to its OpenAI-compatible base URL ending in `/v1`.
2. Run `python -m chatbot.server` from the project root. No third-party Python packages are needed.
3. Run `python -m unittest discover -s tests -v` for local tests. Tests use fakes and a local HTTP server; they do not call paid services.

To verify the live profile endpoint before configuring Redis or search, set `CAREER_API_BASE_URL` and `CAREER_API_USER_CONTEXT_TOKEN`, then run `python -m chatbot.verify_context --email you@example.com`. This service-only check calls `/userContext` directly and prints a redacted summary of the response and projected model context. It does not prove that the frontend proxy forwards the student's session cookie; that needs a separate end-to-end chat test.

You can also build the included Dockerfile for a container host that supports long-lived HTTP responses. Point `CAREER_API_BASE_URL` to the deployed `careerAPI`; configure the frontend's first-party `/chat/*` proxy to forward the `careerai_session` cookie to this service. `ALLOWED_ORIGINS` is a comma-separated list of browser origins permitted to POST; a trusted proxy may omit `Origin`.

## Deploy on Vercel

`pyproject.toml` points Vercel at the WSGI app in `chatbot/wsgi.py`, which serves the same routes as `chatbot.server`. Import the repository with the Python preset, root directory `./`, and the default build and install settings. Add the environment variables from `.env.example` in the project settings; `PORT` is not used. Upstash is required on Vercel because serverless instances do not share memory; without it the service returns `server_misconfigured` and logs the reason. Configuration is read on the first request, so redeploy after changing variables. Point the frontend's `/chat/*` proxy at the deployment URL so the `careerai_session` cookie is forwarded.

## External contracts

- On the first message of a login session, `GET /me` with the forwarded student cookie supplies the authenticated student's email, and `GET /me/access` confirms paid access. The frontend does not supply an email for profile selection.
- Then `GET /userContext?email=<URL-encoded authenticated email>` uses `Authorization: Bearer <CAREER_API_USER_CONTEXT_TOKEN>` from the server environment. The returned `email` must match the authenticated email. The response may contain a null `profile`, root-level `top_careers`, and `contest_scores`. Only first name, available onboarding details, useful career-match fields, and summaries of up to three attempted contests enter the model context. Email, full name, static salary estimates, contest IDs, and timestamps are excluded.
- `GET /me/access` must return `{ "has_access": true | false }` for an authenticated student. It is not called again while the session's snapshot is cached.
- The frontend creates a UUID for each page conversation, sends completed prior turns with every request, and discards an unfinished assistant turn if the stream errors or is cancelled.

## HTTP API

`GET /health` returns `{ "ok": true }` without private configuration.

`POST /chat/messages` accepts JSON:

```json
{
  "conversation_id": "28edc011-a95f-49f4-a9d7-f0e3cc91769d",
  "message": "Which stream is right for me?",
  "history": [
    { "role": "user", "content": "What does an architect do?" },
    { "role": "assistant", "content": "Architects design buildings..." }
  ]
}
```

`history` must alternate user and assistant turns and contain at most six complete exchanges. The message limit is 2,000 characters; history limit is 12,000 characters. The request body must be at most 32 KB. A first request has empty history.

A successful response is `text/event-stream` with `sources` (when web search found sources), repeated `delta`, then `done` events. Each event's `data:` line contains JSON. A failure before streaming returns JSON `{ "error": { "code": "...", "message": "..." } }` with an HTTP status. A failure during streaming sends an `error` event and no `done`. The frontend should render `delta.text` as text or sanitised Markdown, never as raw HTML. Clickable source URLs come only from the `sources` event.

The service keeps one snapshot per login session, frozen from the session's first message: later messages, including new chats, make no careerAPI calls. A changed profile, lost paid access, or logout takes effect when the snapshot expires or the student logs in again with a new session. If the snapshot is gone when a message arrives, the service fetches it again. Older deployments returned `conversation_expired` (HTTP 409) in that case; the frontend still handles it by starting a new chat.

## Search, cache, and privacy

When configured, Brave LLM Context API supplies short web evidence for questions about changing facts. When no useful evidence is available, the service gives a clear verification failure instead of asking the model to guess. Search requests include the user's redacted question and India as the search country; they do not include the full profile. The model is instructed to cite retrieved items as `[1]`, `[2]`, etc. This is web RAG; no embeddings or vector database are used.

The selected cache holds only the projected profile, the careerAPI student ID (for the daily cap), and its creation time. Keys are HMACs of the session cookie, so each login session has one key however many chats it opens. Keys expire 24 hours after the session's first message; reads do not renew them. The in-memory cache is lost on restart and is not shared across processes or instances. For deployment without Upstash, run one process and one instance; after a restart, the next message fetches the profile again. Each student gets 25 completed replies per day, reset at midnight IST; cancelled or failed replies and the verification-unavailable reply do not count. Rate limiting and this daily cap are also per process in this mode. This expiry bounds how long a stale profile or lapsed access can be used. The service never logs cookies, prompts, chat messages, profile values, or search snippets.

`CAREER_API_USER_CONTEXT_TOKEN`, optional Upstash credentials, the selected LLM key, and optional `BRAVE_SEARCH_API_KEY` must remain server-side. If Brave is unset, questions requiring current facts get a verification-unavailable response without calling the model. Configured Redis or careerAPI downtime stops a chat request safely. LLM, Brave Search, hosting, and Redis can each have separate charges.
