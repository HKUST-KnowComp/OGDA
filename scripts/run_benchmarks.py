#!/usr/bin/env python3
"""Run OGDA ontology construction, benchmark tasks and native evaluation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ogda.adaptor.suite import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
