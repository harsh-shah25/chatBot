"""MCP server for browsing the chatBot Upstash Redis (Vercel KV) store over its REST API.

Credentials come from the process environment, then tools/kv_mcp/.env, then the
project .env. Either UPSTASH_REDIS_REST_URL/TOKEN or KV_REST_API_URL/TOKEN work.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

HERE = Path(__file__).resolve().parent
ENV_FILES = (HERE / ".env", HERE.parent.parent / ".env")
URL_KEYS = ("UPSTASH_REDIS_REST_URL", "KV_REST_API_URL")
TOKEN_KEYS = ("UPSTASH_REDIS_REST_TOKEN", "KV_REST_API_TOKEN")

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)

server = MCPServer(
    "upstash-kv",
    instructions="Browse and edit the chatBot Upstash Redis (Vercel KV) database.",
)


def _read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _credentials() -> tuple[str, str]:
    sources = [dict(os.environ), *(_read_env_file(p) for p in ENV_FILES)]
    for source in sources:
        for url_key, token_key in zip(URL_KEYS, TOKEN_KEYS):
            url, token = source.get(url_key, "").strip(), source.get(token_key, "").strip()
            if url and token:
                return url.rstrip("/"), token
    raise ToolError(
        "Upstash credentials not found. Set UPSTASH_REDIS_REST_URL and "
        f"UPSTASH_REDIS_REST_TOKEN in {ENV_FILES[0]}"
    )


def redis(*command: Any) -> Any:
    url, token = _credentials()
    body = json.dumps([str(part) for part in command]).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read() or b"{}")
        if "error" not in payload:
            raise ToolError(f"Upstash HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ToolError(f"Cannot reach Upstash: {exc.reason}") from exc
    if "error" in payload:
        raise ToolError(payload["error"])
    return payload["result"]


def _pairs(flat: list[Any]) -> dict[str, Any]:
    return dict(zip(flat[::2], flat[1::2]))


@server.tool(annotations=READ)
def scan_keys(pattern: str = "*", cursor: str = "0", count: int = 100) -> dict[str, Any]:
    """List keys matching a glob pattern. Pass the returned cursor back to get the next page; "0" means done."""
    next_cursor, keys = redis("SCAN", cursor, "MATCH", pattern, "COUNT", count)
    return {"cursor": str(next_cursor), "keys": sorted(keys)}


@server.tool(annotations=READ)
def get_key(key: str) -> dict[str, Any]:
    """Read a key of any type (string, hash, list, set, zset, json) along with its type and TTL."""
    kind = redis("TYPE", key)
    if kind == "none":
        return {"key": key, "type": "none", "value": None}
    if kind == "string":
        value: Any = redis("GET", key)
        if value[:1] in ("{", "["):
            try:
                value = json.loads(value)
            except ValueError:
                pass
    elif kind == "hash":
        value = _pairs(redis("HGETALL", key))
    elif kind == "list":
        value = redis("LRANGE", key, 0, -1)
    elif kind == "set":
        value = sorted(redis("SMEMBERS", key))
    elif kind == "zset":
        value = _pairs(redis("ZRANGE", key, 0, -1, "WITHSCORES"))
    elif kind in ("ReJSON-RL", "json"):
        value = json.loads(redis("JSON.GET", key, "$"))
    else:
        value = f"<unsupported type {kind}>"
    return {"key": key, "type": kind, "ttl_seconds": redis("TTL", key), "value": value}


@server.tool(annotations=READ)
def db_info() -> dict[str, Any]:
    """Return the number of keys in the database."""
    return {"dbsize": redis("DBSIZE")}


@server.tool(annotations=WRITE)
def set_string(key: str, value: str, ttl_seconds: int | None = None) -> str:
    """Set a string key, optionally with an expiry in seconds."""
    if ttl_seconds:
        return redis("SET", key, value, "EX", ttl_seconds)
    return redis("SET", key, value)


@server.tool(annotations=WRITE)
def expire_key(key: str, ttl_seconds: int) -> bool:
    """Set a key's TTL in seconds; use -1 to remove the expiry (PERSIST). 0 deletes the key."""
    if ttl_seconds < 0:
        return bool(redis("PERSIST", key))
    return bool(redis("EXPIRE", key, ttl_seconds))


@server.tool(annotations=DESTRUCTIVE)
def delete_keys(keys: list[str]) -> int:
    """Delete one or more keys. Returns how many were removed."""
    return redis("DEL", *keys)


@server.tool(annotations=DESTRUCTIVE)
def run_command(command: list[str]) -> Any:
    """Run any raw Redis command, e.g. ["HGET", "user:1", "name"].

    Unrestricted: this can run destructive commands such as FLUSHDB or FLUSHALL.
    """
    return redis(*command)


if __name__ == "__main__":
    server.run()
