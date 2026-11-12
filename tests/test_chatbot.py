import io
import json
import threading
import time
import unittest
import uuid
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from dataclasses import replace
from unittest.mock import patch

from chatbot.core import (ChatError, context_key, daily_key, get_session_cookie, needs_search,
                          normalize_context, safe_source, search_query, validate_message)
from chatbot.providers import HostedProvider
from chatbot.server import ChatService, Config, make_handler
from chatbot.integrations import BraveSearch, CareerApi, MemoryCache, UpstashCache
from chatbot.verify_context import summarize


class FakeCareerApi:
    def __init__(self):
        self.me_calls = 0
        self.context_calls = 0
        self.access_calls = 0
        self.access = True
        self.city = "Pune"

    def student(self, cookie):
        self.me_calls += 1
        return {"id": f"id-{cookie}", "email": f"{cookie}@example.com"}

    def context(self, cookie):
        return self.context_by_email(self.student(cookie)["email"])

    def context_by_email(self, email):
        self.context_calls += 1
        return {"student": {"full_name": "Ananya Sharma", "email": "private@example.com",
                            "current_career": "Architect"},
                "profile": {"class": 10, "city": self.city, "hobbies": ["Drawing"]},
                "latest_quiz": {"riasec_code": "AIC", "results": {
                    "stream_recommendation": "PCM", "top_careers": [
                        {"title": "Architect", "match_pct": 91, "email": "hidden@example.com"}]}}}

    def check_access(self, cookie):
        self.access_calls += 1
        if not self.access:
            raise ChatError(402, "payment_required", "An active plan is required.")


class FakeCache:
    def __init__(self):
        self.values = {}
        self.counts = {}

    def get(self, key):
        return self.values.get(key)

    def put(self, key, value):
        self.values[key] = value

    def rate(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def usage(self, key):
        return self.counts.get(key, 0)

    def add_usage(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1


class FakeSearch:
    def __init__(self):
        self.calls = []
        self.results = [{"title": "Official site", "url": "https://example.org/exam",
                         "date": "2026-10-01", "snippet": "Applications open in November."}]

    def search(self, message):
        self.calls.append(message)
        return self.results


class FakeProvider:
    def __init__(self):
        self.calls = []

    def open_stream(self, prompt, history, message):
        self.calls.append((prompt, history, message))
        return io.BytesIO(b"irrelevant")

    def chunks(self, response):
        response.close()
        yield "A useful answer."


def config():
    return Config(0, frozenset({"https://app.example"}), "http://api.example/",
                  "service-token", "https://redis.example", "token", "s" * 40, "groq", "test-model",
                  "test-key", "", "search-key")


class CoreTests(unittest.TestCase):
    def test_validation_and_cookie(self):
        cid = str(uuid.uuid4())
        self.assertEqual(validate_message({"conversation_id": cid, "message": " Hi ", "history": []})[1], "Hi")
        self.assertEqual(get_session_cookie("other=x; careerai_session=abc123456789012345"), "abc123456789012345")
        self.assertIsNone(get_session_cookie("careerai_session=short"))
        with self.assertRaises(ChatError):
            validate_message({"conversation_id": cid, "message": "hi", "history": [{"role": "assistant", "content": "x"}]})

    def test_profile_projection_excludes_identity_and_nested_extra_fields(self):
        context = normalize_context(FakeCareerApi().context("session"))
        text = json.dumps(context)
        self.assertIn("Ananya", text)
        self.assertIn("Architect", text)
        self.assertNotIn("Sharma", text)
        self.assertNotIn("private@example.com", text)
        self.assertNotIn("hidden@example.com", text)

    def test_actual_user_context_shape_with_null_profile(self):
        raw = {"email": "student@example.com", "full_name": "Asha Rao", "profile": None,
               "top_careers": [{"rank": 1, "title": "Aerospace Engineer", "field": "Engineering",
                                "match_pct": 98, "why": "Likes designing aircraft", "stream": "PCM",
                                "salary": "Stale salary", "emoji": "plane"}],
               "contest_scores": [{"contest": {"id": 18}, "attempted": True, "score": 50,
                                   "correct": 5, "total_questions": 5, "time_taken_s": 180,
                                   "global": {"rank": 2, "total": 3}}]}
        context = normalize_context(raw)
        self.assertEqual(context["student"]["first_name"], "Asha")
        self.assertEqual(context["profile"], {})
        self.assertEqual(context["assessment"]["top_careers"][0]["match_pct"], 98)
        self.assertEqual(context["recent_contests"][0]["correct"], 5)
        text = json.dumps(context)
        for excluded in ("student@example.com", "Rao", "Stale salary", '"contest":', '"id":'):
            self.assertNotIn(excluded, text)
        report = summarize(raw, context)
        self.assertTrue(report["endpoint_reached"])
        self.assertFalse(report["profile_present"])
        self.assertEqual(report["top_careers"][0]["title"], "Aerospace Engineer")
        self.assertNotIn("student@example.com", json.dumps(report))

    def test_search_and_sources(self):
        self.assertTrue(needs_search("What is the latest NATA exam date?"))
        self.assertFalse(needs_search("What does an architect do?"))
        self.assertFalse(needs_search("Which college course gives the best jobs and salary?"))
        self.assertNotIn("student@example.com", search_query("Fees for student@example.com in Pune"))
        self.assertIsNone(safe_source("http://localhost/private"))
        self.assertIsNone(safe_source("https://[::1]/private"))
        self.assertIsNone(safe_source("https://[broken"))

    def test_provider_sse_shapes(self):
        from chatbot.providers import sse_data
        data = io.BytesIO(b'data: {"x": 1}\r\n\r\ndata: [DONE]\n\n')
        self.assertEqual(list(sse_data(data)), ['{"x": 1}', "[DONE]"])
        provider = HostedProvider("groq", "model", "key")
        stream = io.BytesIO(b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n')
        self.assertEqual(list(provider.chunks(stream)), ["Hello"])
        gemini = HostedProvider("gemini", "model", "key")
        gemini_stream = io.BytesIO(b'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\n\n')
        self.assertEqual(list(gemini.chunks(gemini_stream)), ["Hi"])

    def test_optional_configuration_and_memory_cache(self):
        required = {
            "ALLOWED_ORIGINS": "https://app.example",
            "CAREER_API_BASE_URL": "https://api.example",
            "CAREER_API_USER_CONTEXT_TOKEN": "service-token",
            "CACHE_HMAC_SECRET": "s" * 40,
            "LLM_PROVIDER": "gemini",
            "LLM_MODEL": "gemini-3.1-flash-lite",
            "GEMINI_API_KEY": "test-key",
        }
        with patch.dict("os.environ", required, clear=True):
            settings = Config.from_env()
        self.assertEqual((settings.redis_url, settings.redis_token, settings.search_api_key),
                         ("", "", ""))
        with patch.dict("os.environ", {**required, "UPSTASH_REDIS_REST_URL": "https://redis.example"},
                        clear=True):
            with self.assertRaisesRegex(ValueError, "both Upstash credentials"):
                Config.from_env()
        with patch.dict("os.environ", {**required, "VERCEL": "1"}, clear=True):
            with self.assertRaisesRegex(ValueError, "required on Vercel"):
                Config.from_env()

        cache = MemoryCache()
        with patch("chatbot.integrations.time.monotonic", return_value=100):
            cache.put("profile", {"name": "Asha"})
            cache.put("profile", {"name": "Other"})
            self.assertEqual(cache.get("profile"), {"name": "Other"})
            self.assertEqual(cache.rate("rate"), 1)
            self.assertEqual(cache.rate("rate"), 2)
        with patch("chatbot.integrations.time.monotonic", return_value=200):
            self.assertEqual(cache.get("profile"), {"name": "Other"})
        with patch("chatbot.integrations.time.monotonic", return_value=220):
            self.assertEqual(cache.rate("rate"), 1)
            self.assertEqual(cache.usage("daily"), 0)
            cache.add_usage("daily")
            cache.add_usage("daily")
            self.assertEqual(cache.usage("daily"), 2)
        with patch("chatbot.integrations.time.monotonic", return_value=86601):
            self.assertIsNone(cache.get("profile"))
        with patch("chatbot.integrations.time.monotonic", return_value=220 + 172800):
            self.assertEqual(cache.usage("daily"), 0)

    def test_external_contract_clients(self):
        seen = []
        def fake_request(url, **kwargs):
            seen.append((url, kwargs))
            if url.endswith("me/access"):
                return {"has_access": True}
            if url.endswith("/me"):
                return {"student": {"id": 7, "email": "student@example.com"}}
            if "userContext?email=" in url:
                return {"email": "student@example.com", "profile": {"class": 10}}
            if url == "https://redis.example":
                return {"result": "OK"}
            return {"grounding": {"generic": [{"url": "https://example.org/exam",
                                                  "title": "Exam", "snippets": ["Current date"]}]},
                    "sources": {"https://example.org/exam": {"age": ["", "2026-10-23"]}}}
        with patch("chatbot.integrations.json_request", side_effect=fake_request):
            api = CareerApi("https://api.example/", "service-token")
            self.assertEqual(api.context("token")["profile"]["class"], 10)
            self.assertEqual(api.student("token"), {"id": "7", "email": "student@example.com"})
            api.check_access("token")
            UpstashCache("https://redis.example", "secret").put("key", {"profile": {}})
            self.assertEqual(BraveSearch("key").search("Latest exam date?")[0]["date"], "2026-10-23")
        self.assertIn("Cookie", seen[0][1]["headers"])
        self.assertNotIn("Authorization", seen[0][1]["headers"])
        self.assertIn("Bearer", seen[2][1]["headers"]["Authorization"])
        self.assertIn("email=student%40example.com", seen[2][0])
        self.assertEqual(seen[5][1]["body"][0], "SET")
        self.assertEqual(seen[5][1]["body"][-2:], ["EX", 86400])

    def test_context_identity_and_paid_access_are_enforced(self):
        calls = []
        def mismatch(url, **kwargs):
            calls.append(url)
            if url.endswith("/me"):
                return {"student": {"id": 7, "email": "student@example.com"}}
            if url.endswith("/me/access"):
                return {"has_access": True}
            return {"email": "other@example.com", "profile": None}
        with patch("chatbot.integrations.json_request", side_effect=mismatch):
            with self.assertRaises(ChatError) as caught:
                CareerApi("https://api.example", "service-token").context("session")
        self.assertEqual(caught.exception.code, "invalid_context")
        calls.clear()
        def unpaid(url, **kwargs):
            calls.append(url)
            return {"student": {"id": 7, "email": "student@example.com"}} if url.endswith("/me") else {"has_access": False}
        with patch("chatbot.integrations.json_request", side_effect=unpaid):
            with self.assertRaises(ChatError) as caught:
                CareerApi("https://api.example", "service-token").context("session")
        self.assertEqual(caught.exception.status, 402)
        self.assertEqual(len(calls), 2)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeCareerApi()
        self.cache = FakeCache()
        self.search = FakeSearch()
        self.provider = FakeProvider()
        self.service = ChatService(config(), career_api=self.api, cache=self.cache,
                                   search=self.search, provider=self.provider)
        self.cid = str(uuid.uuid4())

    def request(self, message="What does an architect do?", history=None, cookie="session-a"):
        return self.service.prepare(cookie, {"conversation_id": self.cid, "message": message,
                                             "history": history or []})

    HISTORY = [{"role": "user", "content": "What does an architect do?"},
               {"role": "assistant", "content": "A useful answer."}]

    def test_first_message_and_cached_followup(self):
        chunks, sources, searched = self.request()
        self.assertEqual(list(chunks), ["A useful answer."])
        self.assertFalse(searched)
        self.request(message="How do I prepare?", history=self.HISTORY)
        # Only the session's first message calls careerAPI.
        self.assertEqual((self.api.me_calls, self.api.access_calls, self.api.context_calls),
                         (1, 1, 1))

    def test_snapshot_is_frozen_for_the_session_across_new_chats(self):
        self.request()
        self.api.city = "Mumbai"
        self.cid = str(uuid.uuid4())
        self.request()
        self.request(message="Follow-up", history=self.HISTORY)
        self.assertEqual((self.api.me_calls, self.api.access_calls, self.api.context_calls),
                         (1, 1, 1))
        self.assertEqual(len(self.cache.values), 1)
        self.assertIn("Pune", self.provider.calls[-1][0])
        self.assertNotIn("Mumbai", self.provider.calls[-1][0])
        # A new login session takes a fresh snapshot.
        self.request(cookie="session-a2")
        self.assertIn("Mumbai", self.provider.calls[-1][0])

    def test_search_sources_and_no_verification_fallback(self):
        chunks, sources, searched = self.request("Latest NATA exam date?")
        self.assertTrue(searched)
        self.assertEqual(sources[0]["url"], "https://example.org/exam")
        self.assertIn("[1]", self.provider.calls[-1][0])
        self.search.results = []
        self.provider.calls.clear()
        chunks, sources, _ = self.request("Latest NATA exam date?")
        self.assertIn("couldn't verify", next(chunks))
        self.assertEqual(self.provider.calls, [])

    def test_access_is_checked_once_per_session_snapshot(self):
        self.request()
        self.api.access = False
        self.request(message="Follow-up", history=self.HISTORY)
        with self.assertRaisesRegex(ChatError, "active plan"):
            self.request(cookie="session-b")
        # After the snapshot expires or is evicted, the next message checks again.
        self.cache.values.pop(context_key(config().hmac_secret, "session-a"))
        with self.assertRaisesRegex(ChatError, "active plan"):
            self.request(message="Follow-up", history=self.HISTORY)
        self.api.access = True
        self.request(message="Follow-up", history=self.HISTORY)
        self.assertEqual(self.api.context_calls, 2)

    def test_daily_cap_counts_only_completed_model_replies(self):
        usage_key = daily_key(config().hmac_secret, "id-session-a", time.time())
        self.cache.rate = lambda key: 1  # keep the per-minute limit out of the way
        # Unconsumed (cancelled) replies and the canned no-facts reply are not counted.
        self.request()
        self.search.results = []
        self.assertIn("couldn't verify", next(self.request("Latest NATA exam date?")[0]))
        self.assertEqual(self.cache.counts.get(usage_key, 0), 0)
        for _ in range(25):
            self.assertEqual(list(self.request()[0]), ["A useful answer."])
        self.assertEqual(self.cache.counts[usage_key], 25)
        calls = len(self.provider.calls)
        with self.assertRaises(ChatError) as caught:
            self.request()
        self.assertEqual((caught.exception.status, caught.exception.code), (429, "daily_limit_reached"))
        self.assertEqual(len(self.provider.calls), calls)
        # The cap is per student, and a new IST day starts a fresh count.
        self.assertEqual(list(self.request(cookie="session-b")[0]), ["A useful answer."])
        with patch("chatbot.server.time.time", return_value=time.time() + 86400):
            self.assertEqual(list(self.request()[0]), ["A useful answer."])

    def test_daily_key_rolls_over_at_ist_midnight(self):
        secret = config().hmac_secret
        before = daily_key(secret, "7", 1792866599)  # 2026-10-24 23:59:59 IST
        after = daily_key(secret, "7", 1792866600)   # 2026-10-25 00:00:00 IST
        self.assertNotEqual(before, after)
        self.assertEqual(before, daily_key(secret, "7", 1792780200))  # 2026-10-24 00:00:00 IST

    def test_users_and_conversations_are_isolated(self):
        self.request(cookie="session-a")
        self.request(cookie="session-b")
        self.assertEqual(self.api.context_calls, 2)
        self.assertEqual(len(self.cache.values), 2)

    def test_optional_services_use_memory_and_skip_search(self):
        api = FakeCareerApi()
        provider = FakeProvider()
        settings = replace(config(), redis_url="", redis_token="", search_api_key="")
        service = ChatService(settings, career_api=api, provider=provider)
        self.assertIsInstance(service.cache, MemoryCache)
        self.assertIsNone(service.search)
        cid = str(uuid.uuid4())
        request = {"conversation_id": cid, "message": "What does an architect do?", "history": []}
        self.assertEqual(list(service.prepare("session-a", request)[0]), ["A useful answer."])
        request["history"] = [{"role": "user", "content": "What does an architect do?"},
                              {"role": "assistant", "content": "A useful answer."}]
        request["message"] = "How should I prepare?"
        self.assertEqual(list(service.prepare("session-a", request)[0]), ["A useful answer."])
        self.assertEqual((api.me_calls, api.access_calls, api.context_calls), (1, 1, 1))
        provider.calls.clear()
        request["message"] = "What is the latest exam date?"
        chunks, sources, searched = service.prepare("session-a", request)
        self.assertTrue(searched)
        self.assertEqual(sources, [])
        self.assertIn("couldn't verify", next(chunks))
        self.assertEqual(provider.calls, [])

    def test_http_stream_and_origin_gate(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.service))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            body = json.dumps({"conversation_id": self.cid, "message": "What does an architect do?", "history": []})
            connection.request("POST", "/chat/messages", body=body,
                               headers={"Content-Type": "application/json", "Origin": "https://app.example",
                                        "Cookie": "careerai_session=session-token-abcdef"})
            response = connection.getresponse()
            text = response.read().decode()
            self.assertEqual(response.status, 200)
            self.assertIn("event: delta", text)
            self.assertIn("event: done", text)
            connection.close()
            connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            connection.request("POST", "/chat/messages", body=body,
                               headers={"Content-Type": "application/json", "Origin": "https://evil.example",
                                        "Cookie": "careerai_session=session-token-abcdef"})
            self.assertEqual(connection.getresponse().status, 403)
            connection.close()
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)


    def wsgi_call(self, origin):
        from wsgiref.util import setup_testing_defaults
        from chatbot import wsgi
        body = json.dumps({"conversation_id": self.cid, "message": "What does an architect do?",
                           "history": []}).encode()
        environ = {"REQUEST_METHOD": "POST", "PATH_INFO": "/chat/messages",
                   "CONTENT_TYPE": "application/json", "CONTENT_LENGTH": str(len(body)),
                   "HTTP_ORIGIN": origin, "HTTP_COOKIE": "careerai_session=session-token-abcdef",
                   "wsgi.input": io.BytesIO(body)}
        setup_testing_defaults(environ)
        seen = {}
        def start_response(status, headers):
            seen["status"], seen["headers"] = status, dict(headers)
        with patch.object(wsgi, "_service", self.service):
            text = b"".join(wsgi.app(environ, start_response)).decode()
        return seen, text

    def test_wsgi_stream_and_origin_gate(self):
        seen, text = self.wsgi_call("https://app.example")
        self.assertEqual(seen["status"], "200 OK")
        self.assertEqual(seen["headers"]["Content-Type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: delta", text)
        self.assertIn("event: done", text)
        seen, _ = self.wsgi_call("https://evil.example")
        self.assertEqual(seen["status"], "403 Forbidden")

    def test_wsgi_reports_missing_configuration(self):
        from wsgiref.util import setup_testing_defaults
        from chatbot import wsgi
        environ = {"PATH_INFO": "/health"}
        setup_testing_defaults(environ)
        seen = {}
        with patch.object(wsgi, "_service", None), patch.object(wsgi, "load_dotenv"), \
                patch.dict("os.environ", {}, clear=True), self.assertLogs("chatbot", "ERROR"):
            body = b"".join(wsgi.app(environ, lambda status, headers: seen.update(status=status)))
        self.assertEqual(seen["status"], "500 Internal Server Error")
        self.assertIn(b"server_misconfigured", body)


if __name__ == "__main__":
    unittest.main()
