# ──────────────────────────────────────────────────────────────────────────────
# File: src/tariffs.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import json

def load_tariff(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


