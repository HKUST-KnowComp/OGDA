#!/usr/bin/env python3
"""Prepare verified benchmark data copies for OGDA ontology construction."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogda.adaptor.prepare_datalakes import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
