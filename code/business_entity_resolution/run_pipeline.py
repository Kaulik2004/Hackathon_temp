#!/usr/bin/env python3
"""Convenience entry point for the pipeline.

Run this from anywhere; it fixes the working directory to
``student_resource/`` (two levels up from ``code/business_entity_resolution/``)
so the pipeline's default relative paths (``dataset/``, ``output/``) match
``utils/validate_submission.py``'s own defaults, and puts this directory on
``sys.path`` so ``src`` is importable as a package.

Equivalent to running, from ``student_resource/``:
    python -m src.pipeline <args...>
with PYTHONPATH including code/business_entity_resolution/.
"""

import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # code/business_entity_resolution/
_STUDENT_RESOURCE_DIR = _THIS_DIR.parent.parent  # student_resource/

sys.path.insert(0, str(_THIS_DIR))
os.chdir(_STUDENT_RESOURCE_DIR)

from src.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
