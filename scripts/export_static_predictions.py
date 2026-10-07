#!/usr/bin/env python3
"""Generate data/static_predictions.json from the API /dashboard endpoint.

This freeze of predictions is shipped with the Vercel deployment so the
site works without a backing database / network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app
from fastapi.testclient import TestClient


def main() -> None:
    client = TestClient(app)
    resp = client.get("/dashboard")
    if resp.status_code != 200:
        print(f"[export_static] dashboard returned {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    out = Path("data/static_predictions.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":"), default=str), encoding="utf-8")
    print(f"[export_static] wrote {data.get('count', 0)} predictions to {out}")


if __name__ == "__main__":
    main()
