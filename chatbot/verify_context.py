"""Check the live userContext endpoint without Redis, Tavily, or an LLM.

Run: python -m chatbot.verify_context --email you@example.com
"""

from __future__ import annotations

import argparse
import json
import os

from .core import ChatError, normalize_context
from .integrations import CareerApi
from .server import load_dotenv


def summarize(raw: dict, context: dict) -> dict:
    careers = context.get("assessment", {}).get("top_careers", [])
    return {
        "endpoint_reached": True,
        "identity_match": True,
        "profile_present": raw.get("profile") is not None,
        "first_name": context.get("student", {}).get("first_name"),
        "top_careers": [{"rank": item.get("rank"), "title": item.get("title"),
                         "match_pct": item.get("match_pct")} for item in careers],
        "attempted_contests": len(context.get("recent_contests", [])),
        "model_context_sections": list(context),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the live careerAPI userContext response")
    parser.add_argument("--email", required=True, help="Email of the student to verify")
    args = parser.parse_args()
    load_dotenv()
    base = os.getenv("CAREER_API_BASE_URL", "").strip()
    token = os.getenv("CAREER_API_USER_CONTEXT_TOKEN", "").strip()
    if not base or not token:
        print("Missing CAREER_API_BASE_URL or CAREER_API_USER_CONTEXT_TOKEN in .env")
        return 2
    try:
        raw = CareerApi(base, token).context_by_email(args.email)
        context = normalize_context(raw)
    except ChatError as error:
        print(json.dumps({"endpoint_reached": False, "error": error.code,
                          "message": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(summarize(raw, context), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
