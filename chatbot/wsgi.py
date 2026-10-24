"""WSGI entry point for Vercel (configured in pyproject.toml as chatbot.wsgi:app)."""

from __future__ import annotations

import logging
import threading
from http import HTTPStatus

from .server import LOG, ChatService, Config, handle, json_response, load_dotenv


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_service: ChatService | None = None
_lock = threading.Lock()


def get_service() -> ChatService:
    # Built on the first request so a missing variable returns a logged 500
    # instead of failing the import.
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                load_dotenv()
                _service = ChatService(Config.from_env())
    return _service


def _header(environ: dict, name: str) -> str | None:
    key = name.upper().replace("-", "_")
    if key not in {"CONTENT_TYPE", "CONTENT_LENGTH"}:
        key = "HTTP_" + key
    return environ.get(key)


def app(environ, start_response):
    try:
        service = get_service()
    except ValueError as error:
        LOG.error("configuration_error=%s", error)
        status, headers, body = json_response(
            500, {"error": {"code": "server_misconfigured", "message": "Please try again later."}})
    else:
        status, headers, body = handle(service, environ.get("REQUEST_METHOD", "GET"),
                                       environ.get("PATH_INFO") or "/",
                                       lambda name: _header(environ, name),
                                       environ["wsgi.input"].read)
    start_response(f"{status} {HTTPStatus(status).phrase}", headers)
    return body
