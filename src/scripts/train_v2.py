"""Module wrapper for the canonical training CLI in ``scripts.train_v2``."""

from __future__ import annotations

from scripts.train_v2 import *  # noqa: F401,F403
from scripts.train_v2 import _save_run_metadata, _set_seed, main  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main())
