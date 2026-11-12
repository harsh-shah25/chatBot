"""careerAPI, Upstash Redis REST, and Brave Search clients."""

from __future__ import annotations

import json
import re
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from .core import ChatError, safe_source, search_query


def json_request(url: str, *, method: str = "GET", headers: dict | None = None,
                 body: object = None, timeout: int = 10) -> object:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(url, data=payload, method=method,
                      headers={"Accept": "application/json", **(headers or {})})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError:
        raise
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        raise ChatError(502, "upstream_unavailable", "An upstream service is unavailable.") from error


class CareerApi:
    def __init__(self, base_url: str, user_context_token: str):
        self.base_url = base_url.rstrip("/") + "/"
        self.user_context_token = user_context_token

    def _get(self, path: str, *, cookie: str | None = None, bearer: str | None = None) -> dict:
        headers = {}
        if cookie:
            headers["Cookie"] = f"careerai_session={cookie}"
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        try:
            result = json_request(urljoin(self.base_url, path.lstrip("/")),
                                  headers=headers)
        except HTTPError as error:
            if error.code == 401:
                raise ChatError(401, "unauthenticated", "Please log in.") from None
            if error.code == 402:
                raise ChatError(402, "payment_required", "An active plan is required for AI Chat.") from None
            raise ChatError(502, "career_api_unavailable", "Could not check your account.") from error
        if not isinstance(result, dict):
            raise ChatError(502, "invalid_context", "careerAPI returned invalid data.")
        return result

    def student(self, cookie: str) -> dict:
        """The logged-in student's id and email, from GET /me."""
        student = self._get("me", cookie=cookie).get("student")
        student = student if isinstance(student, dict) else {}
        email = student.get("email")
        if student.get("id") is None or not isinstance(email, str) or not email.strip():
            raise ChatError(502, "invalid_student", "Could not identify your account.")
        return {"id": str(student["id"]), "email": email}

    def context(self, cookie: str) -> dict:
        email = self.student(cookie)["email"]
        self.check_access(cookie)
        return self.context_by_email(email)

    def context_by_email(self, email: str) -> dict:
        """Service-only lookup; pass only an email that came from the caller's own /me."""
        if not isinstance(email, str) or not email.strip():
            raise ChatError(400, "invalid_email", "An email is required for this check.")
        path = "userContext?" + urlencode({"email": email.strip()})
        context = self._get(path, bearer=self.user_context_token)
        returned_email = context.get("email")
        if not isinstance(returned_email, str) or returned_email.casefold() != email.strip().casefold():
            raise ChatError(502, "invalid_context", "careerAPI returned the wrong student profile.")
        return context

    def check_access(self, cookie: str) -> None:
        access = self._get("me/access", cookie=cookie)
        if access.get("has_access") is not True:
            raise ChatError(402, "payment_required", "An active plan is required for AI Chat.")


class MemoryCache:
    """Process-local profile cache with the same expiry and rate API as Upstash."""

    PROFILE_TTL = 86400
    RATE_TTL = 120
    USAGE_TTL = 172800

    def __init__(self):
        self._profiles = {}
        self._rates = {}
        self._usage = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            entry = self._profiles.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at <= time.monotonic():
                del self._profiles[key]
                return None
            return value

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._profiles[key] = (value, time.monotonic() + self.PROFILE_TTL)

    def rate(self, key: str) -> int:
        with self._lock:
            now = time.monotonic()
            count, expires_at = self._rates.get(key, (0, 0))
            if expires_at <= now:
                count, expires_at = 0, now + self.RATE_TTL
            count += 1
            self._rates[key] = (count, expires_at)
            # Rate keys rotate every minute; remove expired windows as they accumulate.
            if len(self._rates) > 1000:
                self._rates = {k: v for k, v in self._rates.items() if v[1] > now}
            return count

    def usage(self, key: str) -> int:
        with self._lock:
            count, expires_at = self._usage.get(key, (0, 0))
            return count if expires_at > time.monotonic() else 0

    def add_usage(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            count, expires_at = self._usage.get(key, (0, 0))
            if expires_at <= now:
                count, expires_at = 0, now + self.USAGE_TTL
            self._usage[key] = (count + 1, expires_at)
            # Daily keys rotate at midnight; drop finished days as they accumulate.
            if len(self._usage) > 1000:
                self._usage = {k: v for k, v in self._usage.items() if v[1] > now}


class UpstashCache:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token

    def _post(self, url: str, body: list) -> object:
        try:
            return json_request(url, method="POST",
                                headers={"Authorization": f"Bearer {self.token}",
                                         "Content-Type": "application/json"},
                                body=body, timeout=5)
        except (HTTPError, ChatError) as error:
            raise ChatError(503, "cache_unavailable", "Chat memory is temporarily unavailable.") from error

    @staticmethod
    def _result(response: object) -> object:
        if not isinstance(response, dict) or response.get("error"):
            raise ChatError(503, "cache_unavailable", "Chat memory is temporarily unavailable.")
        return response.get("result")

    def command(self, *parts) -> object:
        return self._result(self._post(self.url, list(parts)))

    def pipeline(self, *commands: list) -> list:
        """Run several commands in one HTTP round trip."""
        responses = self._post(self.url + "/pipeline", list(commands))
        if not isinstance(responses, list) or len(responses) != len(commands):
            raise ChatError(503, "cache_unavailable", "Chat memory is temporarily unavailable.")
        return [self._result(response) for response in responses]

    def get(self, key: str) -> dict | None:
        # Plain GET: the expiry is fixed from the session's first message, not renewed.
        value = self.command("GET", key)
        if value is None:
            return None
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return data
        except (TypeError, ValueError):
            pass
        raise ChatError(503, "cache_unavailable", "Chat memory is temporarily unavailable.")

    def put(self, key: str, value: dict) -> None:
        self.command("SET", key, json.dumps(value, ensure_ascii=False), "EX", 86400)

    def rate(self, key: str) -> int:
        count, _ = self.pipeline(["INCR", key], ["EXPIRE", key, 120, "NX"])
        return int(count)

    def usage(self, key: str) -> int:
        return int(self.command("GET", key) or 0)

    def add_usage(self, key: str) -> None:
        # Two days, so a key first written just before midnight outlives its own day.
        self.pipeline(["INCR", key], ["EXPIRE", key, 172800, "NX"])


class BraveSearch:
    ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, message: str) -> list[dict]:
        query = search_query(message)
        if not query:
            return []
        try:
            data = json_request(self.ENDPOINT, method="POST", timeout=15,
                                headers={"X-Subscription-Token": self.api_key,
                                         "Content-Type": "application/json"},
                                body={"q": query, "country": "IN", "safesearch": "strict",
                                      "count": 8, "maximum_number_of_urls": 5,
                                      "maximum_number_of_tokens": 2500,
                                      "enable_source_metadata": True})
        except (HTTPError, ChatError) as error:
            raise ChatError(502, "search_unavailable", "Current information could not be checked.") from error
        generic = data.get("grounding", {}).get("generic", []) if isinstance(data, dict) else []
        sources = data.get("sources", {}) if isinstance(data, dict) else {}
        results = []
        for item in generic[:5] if isinstance(generic, list) else []:
            if not isinstance(item, dict):
                continue
            url = safe_source(item.get("url"))
            snippets = item.get("snippets")
            if not url or not isinstance(snippets, list):
                continue
            snippet = " ".join(str(s) for s in snippets[:3])[:1800]
            if not snippet:
                continue
            meta = sources.get(item["url"], {}) if isinstance(sources, dict) else {}
            age = meta.get("age", []) if isinstance(meta, dict) else []
            date = age[1] if isinstance(age, list) and len(age) > 1 and isinstance(age[1], str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", age[1]) else None
            results.append({"title": str(item.get("title") or meta.get("title") or url)[:150],
                            "url": url, "date": date, "snippet": snippet})
        return results
