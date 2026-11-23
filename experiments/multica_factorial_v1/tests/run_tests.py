"""Dependency-free test runner for the Multi-CA core."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_multica_core as tests

names = sorted(name for name in dir(tests) if name.startswith('test_'))
failed = []
for name in names:
    try:
        getattr(tests, name)()
        print('PASS', name)
    except Exception as exc:
        failed.append((name, repr(exc)))
        print('FAIL', name, repr(exc))
if failed:
    print('FAILED', failed)
    raise SystemExit(1)
print('ALL_TESTS_PASS', len(names))
