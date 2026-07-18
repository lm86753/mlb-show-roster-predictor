"""Vercel serverless entry point for the MLB Show roster predictor API."""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
os.environ["VERCEL"] = "1"

from src.api.main import app
