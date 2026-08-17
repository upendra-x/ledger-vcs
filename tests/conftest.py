"""Makes ``tests/support`` importable from any test module.

pytest puts each test file's own directory on the path, not the suite root, so
without this a helper shared by the integration and end-to-end suites would have
to be duplicated or reached by a relative import.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS))
# The project root too, so the suite can import ``demo`` — the demonstration
# is exercised *by* the tests rather than living beside them and rotting.
sys.path.insert(0, str(_TESTS.parent))
