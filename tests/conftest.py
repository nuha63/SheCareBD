"""
conftest.py — adds the backend directory to sys.path
so pytest can import `services.*` when run from the project root.

Usage:
    pytest -q backend/tests/test_safety_filter.py
"""
import sys
import os

# Add backend/ to sys.path
backend_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(backend_dir))
