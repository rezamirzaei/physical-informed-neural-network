"""Repository-local entry point for the Burgers-equation PINN CLI.

Usage::

    python main.py --smoke-test
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from physics_informed_neural_network.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
