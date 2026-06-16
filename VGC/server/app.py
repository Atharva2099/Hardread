from __future__ import annotations

"""
Thin wrapper so `openenv` and `uv run server` can find the ASGI app.
"""

from env.server.app import app as _inner_app
from env.server.app import main as _inner_main

app = _inner_app


def main() -> None:
    _inner_main()


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()


