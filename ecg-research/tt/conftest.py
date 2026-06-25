"""Pytest bootstrap. I put the project root (this file's directory) at the
front of 'sys.path' so the test imports ('from src...' / 'import scripts...')
resolve regardless of where pytest is launched from -- a chronic annoyance
otherwise. It backs up 'pytest.ini''s 'pythonpath = .'. This runs on its own
during pytest collection; there is nothing to call by hand."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
