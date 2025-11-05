# ──────────────────────────────────────────────────────────────────────────────
# File: src/heating.py
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np

def setpoint_by_hour(hour: int, day_c: float, night_c: float) -> float:
    return float(day_c) if 6 <= int(hour) <= 22 else float(night_c)

# Envelope multipliers by class (tunable)
_ENVELOPE_FACTOR = {"good": 0.9, "medium": 1.0, "poor": 1.2}

def ua(envelope_class: str, area_m2: float, ua_per_m2: float) -> float:
    f = _ENVELOPE_FACTOR.get(str(envelope_class).lower(), 1.0)
    return float(ua_per_m2) * float(area_m2) * float(f)


def cop(system: str, tout_c: np.ndarray, a: float, b: float) -> np.ndarray:
    if str(system).lower() == "heat_pump":
        return np.maximum(1.0, a + b * tout_c)
    return np.ones_like(tout_c)


def heating_load_kw(tset_c: np.ndarray, tout_c: np.ndarray, UA: float, COP: np.ndarray | float) -> np.ndarray:
    delta = np.maximum(0.0, tset_c - tout_c)
    return (UA * delta) / np.maximum(1.0, COP)


