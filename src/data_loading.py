# src/data_loading.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import json


# you can call this with either str or Path
def _to_path(p) -> Path:
    return Path(p).expanduser().resolve()


def load_archetypes(path: str | Path) -> pd.DataFrame:
    """Load archetypes.csv"""
    path = _to_path(path)
    df = pd.read_csv(path)
    # optional: ensure required columns exist
    # Archetype, Weight, FloorArea_m2, Envelope_Class, UA_per_m2, HeatSystem, ...
    return df


def load_weather(path: str | Path) -> pd.DataFrame:
    """Load weather_winter_design.csv -> columns: hour, T_out_C"""
    path = _to_path(path)
    df = pd.read_csv(path)
    # make sure hour is int
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype(int)
    return df


def load_baseload_profiles(path: str | Path) -> pd.DataFrame:
    """Load baseload_profiles.csv -> columns: hour, apt, detached, ..."""
    path = _to_path(path)
    df = pd.read_csv(path)
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype(int)
    return df


def load_tariff(path: str | Path) -> dict:
    """Load tariffs_*.json"""
    path = _to_path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
