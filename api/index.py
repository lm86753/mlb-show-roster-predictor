"""Vercel serverless entry point for the MLB Show roster predictor API."""
import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
os.environ["VERCEL"] = "1"

from starlette.routing import Mount
from starlette.applications import Starlette

from src.api.main import app as api_app

app = Starlette(routes=[Mount("/api", api_app)])
