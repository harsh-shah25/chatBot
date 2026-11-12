"""HTTP and SSE entry point. Run with: python -m chatbot.server"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import (ChatError, context_key, daily_key, get_session_cookie, needs_search,
                   normalize_context, rate_key, system_prompt, validate_message)
from .integrations import BraveSearch, CareerApi, MemoryCache, UpstashCache
from .providers import HostedProvider


LOG = logging.getLogger("chatbot")
NO_CURRENT_FACTS = ("I couldn't verify current information right now. Please check an official "
                    "college, exam, or government source and try again later.")
DAILY_LIMIT = 25


@dataclass(frozen=True)
class Config:
    port: int
    allowed_origins: frozenset[str]
    career_api_base_url: str
    user_context_token: str
    redis_url: str
    redis_token: str
    hmac_secret: str
    llm_provider: str
    llm_model: str
    llm_api_key: str
    omniroute_base_url: str
    search_api_key: str

    @classmethod
    def from_env(cls) -> "Config":
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        if provider not in {"groq", "gemini", "omniroute"}:
            raise ValueError("LLM_PROVIDER must be groq, gemini, or omniroute")
        key_name = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
                    "omniroute": "OMNIROUTE_API_KEY"}[provider]
        def required(name):
            value = os.getenv(name, "").strip()
            if not value:
                raise ValueError(f"Missing {name}")
            return value
        secret = required("CACHE_HMAC_SECRET")
        if len(secret) < 32:
            raise ValueError("CACHE_HMAC_SECRET must have at least 32 characters")
        base = required("OMNIROUTE_BASE_URL") if provider == "omniroute" else ""
        redis_url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
        redis_token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
        if bool(redis_url) != bool(redis_token):
            raise ValueError("Set both Upstash credentials or leave both empty")
        if os.getenv("VERCEL") and not redis_url:
            # Serverless instances do not share memory, so follow-up messages would expire.
            raise ValueError("Upstash credentials are required on Vercel")
        return cls(
            port=int(os.getenv("PORT", "3001")),
            allowed_origins=frozenset(s.strip() for s in required("ALLOWED_ORIGINS").split(",") if s.strip()),
            career_api_base_url=required("CAREER_API_BASE_URL"),
            user_context_token=required("CAREER_API_USER_CONTEXT_TOKEN"),
            redis_url=redis_url,
            redis_token=redis_token,
            hmac_secret=secret, llm_provider=provider,
            llm_model=required("LLM_MODEL"), llm_api_key=required(key_name),
            omniroute_base_url=base,
            search_api_key=os.getenv("BRAVE_SEARCH_API_KEY", "").strip(),
        )


def load_dotenv() -> None:
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


class ChatService:
    def __init__(self, config: Config, *, career_api=None, cache=None, search=None, provider=None):
        self.config = config
        self.career_api = career_api or CareerApi(config.career_api_base_url, config.user_context_token)
        self.cache = cache if cache is not None else (
            UpstashCache(config.redis_url, config.redis_token)
            if config.redis_url else MemoryCache()
        )
        self.search = search if search is not None else (
            BraveSearch(config.search_api_key) if config.search_api_key else None
        )
        self.provider = provider or HostedProvider(config.llm_provider, config.llm_model,
                                                    config.llm_api_key, config.omniroute_base_url)
        self._pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="chat-io")

    def _timed(self, timings: dict, name: str, fn, *args):
        started = time.monotonic()
        try:
            return fn(*args)
        finally:
            timings[name] = int((time.monotonic() - started) * 1000)

    def prepare(self, cookie: str, body: object, timings: dict | None = None):
        timings = {} if timings is None else timings
        _, message, history = validate_message(body)
        # The snapshot is frozen per login session: only the first message of a session
        # calls careerAPI (/me, /me/access, userContext). Later messages, in any chat,
        # reuse it until it expires.
        key = context_key(self.config.hmac_secret, cookie)
        rate = self._pool.submit(self._timed, timings, "rate", self.cache.rate,
                                 rate_key(self.config.hmac_secret, cookie, time.time()))
        cached = self._pool.submit(self._timed, timings, "cache_get", self.cache.get, key)
        if rate.result() > 20:
            raise ChatError(429, "rate_limited", "Too many messages. Please wait a minute.")
        record = cached.result()
        if not (isinstance(record, dict) and isinstance(record.get("context"), dict)
                and isinstance(record.get("student_id"), str)):
            record = self._load_session(cookie, key, timings)
        context = record["context"]
        usage_key = daily_key(self.config.hmac_secret, record["student_id"], time.time())
        used = self._timed(timings, "daily", self.cache.usage, usage_key)
        # Checked before search and the model, which both spend quota.
        if used >= DAILY_LIMIT:
            raise ChatError(429, "daily_limit_reached",
                            f"You've used all {DAILY_LIMIT} messages for today. Please come back tomorrow.")
        search_used = needs_search(message)
        evidence = []
        if search_used:
            if self.search is not None:
                try:
                    evidence = self._timed(timings, "search", self.search.search, message)
                except ChatError:
                    pass
            if not evidence:
                return iter((NO_CURRENT_FACTS,)), [], search_used
        prompt = system_prompt(context, evidence)
        response = self._timed(timings, "llm_open", self.provider.open_stream,
                               prompt, history, message)
        sources = [{"title": item["title"], "url": item["url"], "date": item["date"]}
                   for item in evidence]
        return self._counted(self.provider.chunks(response), usage_key), sources, search_used

    def _load_session(self, cookie: str, key: str, timings: dict) -> dict:
        me = self._pool.submit(self._timed, timings, "me", self.career_api.student, cookie)
        access = self._pool.submit(self._timed, timings, "access",
                                   self.career_api.check_access, cookie)
        # Read in a fixed order to keep 401 ahead of 402.
        student = me.result()
        access.result()
        context = normalize_context(self._timed(timings, "user_context",
                                                self.career_api.context_by_email,
                                                student["email"]))
        record = {"student_id": student["id"], "context": context, "created_at": time.time()}
        self._timed(timings, "cache_put", self.cache.put, key, record)
        return record

    def _counted(self, chunks, usage_key: str):
        """Yield the reply, then count it toward the daily cap.

        Only a fully streamed reply is counted: a provider error or client disconnect
        stops the generator before the increment. The canned no-facts reply is not
        wrapped because it never reaches the model."""
        yield from chunks
        try:
            self.cache.add_usage(usage_key)
        except ChatError:
            # The student already has the full answer; don't turn it into a stream error.
            LOG.warning("daily usage increment failed")


def json_response(status: int, body: dict):
    data = json.dumps(body).encode("utf-8")
    return status, [("Content-Type", "application/json; charset=utf-8"),
                    ("Content-Length", str(len(data))), ("Cache-Control", "no-store")], [data]


def sse_event(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


def format_timings(timings: dict) -> str:
    return " ".join(f"{name}_ms={value}" for name, value in timings.items())


def stream_events(service: ChatService, chunks, sources, search_used, request_id, started,
                  timings: dict | None = None):
    """Yield SSE frames. Closing this generator (client disconnect) closes the model stream."""
    timings = {} if timings is None else timings
    try:
        if sources:
            yield sse_event("sources", {"items": sources})
        for text in chunks:
            timings.setdefault("first_token", int((time.monotonic() - started) * 1000))
            yield sse_event("delta", {"text": text})
        yield sse_event("done", {"finish_reason": "stop"})
        LOG.info("request_id=%s status=200 provider=%s search=%s elapsed_ms=%d %s",
                 request_id, service.config.llm_provider, search_used,
                 int((time.monotonic() - started) * 1000), format_timings(timings))
    except ChatError as error:
        LOG.info("request_id=%s status=stream_error error=%s %s", request_id, error.code,
                 format_timings(timings))
        yield sse_event("error", {"code": error.code, "message": str(error)})
    except Exception:
        LOG.error("request_id=%s status=stream_error error=stream_error", request_id)
        yield sse_event("error", {"code": "stream_error", "message": "The reply stopped unexpectedly."})
    finally:
        close = getattr(chunks, "close", None)
        if close:
            close()


def handle(service: ChatService, method: str, path: str, header, read):
    """Transport-neutral routing shared by the stdlib server and the WSGI app.

    Returns (status, headers, body), where body is an iterable of bytes."""
    if method == "GET" and path == "/health":
        return json_response(200, {"ok": True})
    if method != "POST" or path != "/chat/messages":
        return json_response(404, {"error": {"code": "not_found", "message": "No such endpoint."}})
    request_id = uuid.uuid4().hex
    started = time.monotonic()
    timings = {}
    try:
        origin = header("Origin")
        if origin and origin not in service.config.allowed_origins:
            raise ChatError(403, "origin_forbidden", "This origin is not allowed.")
        content_type = (header("Content-Type") or "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ChatError(400, "invalid_content_type", "Send application/json.")
        cookie = get_session_cookie(header("Cookie"))
        if not cookie:
            raise ChatError(401, "unauthenticated", "Please log in.")
        try:
            length = int(header("Content-Length") or "-1")
        except ValueError:
            length = -1
        if length < 1 or length > 32768:
            raise ChatError(400, "invalid_request", "Request body must be at most 32 KB.")
        try:
            body = json.loads(read(length))
        except (ValueError, UnicodeDecodeError):
            raise ChatError(400, "invalid_json", "Request body is not valid JSON.") from None
        chunks, sources, search_used = service.prepare(cookie, body, timings)
    except ChatError as error:
        LOG.info("request_id=%s status=%s error=%s elapsed_ms=%d %s", request_id, error.status,
                 error.code, int((time.monotonic() - started) * 1000), format_timings(timings))
        return json_response(error.status, {"error": {"code": error.code, "message": str(error)}})
    except Exception:
        LOG.error("request_id=%s status=500 error=server_error", request_id)
        return json_response(500, {"error": {"code": "server_error", "message": "Please try again."}})
    headers = [("Content-Type", "text/event-stream; charset=utf-8"),
               ("Cache-Control", "no-store, no-transform"), ("X-Accel-Buffering", "no")]
    return 200, headers, stream_events(service, chunks, sources, search_used, request_id,
                                       started, timings)


def make_handler(service: ChatService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CareerChat/0.1"

        def log_message(self, _format, *_args):
            # BaseHTTPRequestHandler normally logs the URL. Keep logs metadata-only.
            pass

        def _respond(self):
            status, headers, body = handle(service, self.command, self.path,
                                           self.headers.get, self.rfile.read)
            self.send_response(status)
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            try:
                for data in body:
                    self.wfile.write(data)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                close = getattr(body, "close", None)
                if close:
                    close()

        do_GET = do_POST = _respond

    return Handler


def main():
    load_dotenv()
    config = Config.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer(("0.0.0.0", config.port), make_handler(ChatService(config)))
    LOG.info("chatbot listening on port %d", config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
