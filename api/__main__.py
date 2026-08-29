"""``python -m api`` — run the dev server (uvicorn api.app:app --reload)."""
from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code change (dev)")
    args = ap.parse_args()

    import uvicorn
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
