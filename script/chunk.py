#!/usr/bin/env python3
"""Compatibility wrapper: `chunk.py` now forwards to `chunk_ter.py`."""

from __future__ import annotations

import sys

try:
    from .chunk_ter import main
except ImportError:
    from chunk_ter import main


if __name__ == "__main__":
    print("`script/chunk.py` is deprecated; forwarding to `script/chunk_ter.py`.", file=sys.stderr)
    main()
