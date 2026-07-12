"""Module wrapper for the canonical evaluation CLI in ``scripts.evaluate``."""

from __future__ import annotations

from scripts.evaluate import *  # noqa: F401,F403
from scripts.evaluate import main


if __name__ == "__main__":
    raise SystemExit(main())
