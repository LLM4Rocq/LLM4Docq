#!/usr/bin/env python3
"""Minimal HTTP client for the retrieval server."""

from __future__ import annotations

import argparse
import json
from urllib import error, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call retrieval server /search endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010", help="Server base URL.")
    parser.add_argument("--env", required=True, help="Environment name.")
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--k", type=int, default=5, help="Top-k results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise ValueError("--k must be >= 1")

    payload = {
        "query": args.query,
        "env": args.env,
        "k": args.k,
    }
    data = json.dumps(payload).encode("utf-8")
    url = args.base_url.rstrip("/") + "/search"

    req = request.Request(
        url=url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {details}") from exc
    except error.URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
