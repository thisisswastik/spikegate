"""
conftest.py — pytest configuration and shared fixtures.

Sets PYTHONIOENCODING to utf-8 for all test runs (needed on Windows
where the default cp1252 codec rejects Unicode characters in print statements).
"""
import os
import sys

# Force UTF-8 output encoding for all tests — prevents UnicodeEncodeError
# on Windows (cp1252 doesn't handle checkmarks, rupee symbol, etc.)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # If reconfigure fails, the env var above is the fallback
