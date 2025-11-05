# ──────────────────────────────────────────────────────────────────────────────
# File: src/simulate.py
# Simulates hourly energy use and generates household and feeder kpis + tariffs
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import pandas as pd
from .heating import cop, ua, heating_load_kw, setpoint_by_hour


def simulate_day(
    households_df: pd.DataFrame,
    weather_series: pd.Series,
    baseload_profiles: pd.DataFrame,
    tariff: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (sim_long, hh_kpis, feeder_kpis).

    sim_long columns: [hour, meter_id, base_kW, heat_kW, total_kW, price_$perkWh]
    hh_kpis columns: [meter_id, kWh_total, kWh_heat, kWh_base, bill_$]
    feeder_kpis: {"peak_MW": float, "t_peak": int, "daily_MWh": float}
    """
    # make sure hours are integers so reindex(...) works
    hours = weather_series.index.to_numpy(dtype=int)

    # Map baseload profiles to quick lookup (hour → kW)
    bl = baseload_profiles.copy()
    if bl["hour"].dtype != int:
        bl["hour"] = bl["hour"].astype(int)
    bl = bl.set_index("hour")

    # Precompute price by hour from tariff
    price_by_hour = _price_series_from_tariff(hours, tariff)

    sim_rows = []

    for _, row in households_df.iterrows():
        # Setpoint schedule
        tset = np.array(
            [setpoint_by_hour(h, row.Setpoint_Day_C, row.Setpoint_Night_C) for h in hours]
        )

        # Weather
        tout = weather_series.to_numpy()

        # Envelope / COP
        UA = ua(row.Envelope_Class, row.FloorArea_m2, row.UA_per_m2)
        if str(row.HeatSystem).lower() == "heat_pump":
            COP = np.maximum(1.0, row.HP_COP_a + row.HP_COP_b * tout)
        else:
            COP = np.ones_like(tout)

        # Heating
        heat_kW = heating_load_kw(tset, tout, UA, COP)

        # Baseload
        bl_prof_id = str(row.Baseload_Profile_ID)
        if bl_prof_id not in bl.columns:
            raise KeyError(
                f"Baseload profile '{bl_prof_id}' not found in baseload_profiles"
            )
        base_kW = bl[bl_prof_id].reindex(hours).to_numpy()

        # optional per-household baseload multiplier (we set this in app_shiny)
        if "Baseload_Mult" in row.index:
            base_kW = base_kW * float(row.Baseload_Mult)

        total_kW = base_kW + heat_kW

        sim_rows.append(
            pd.DataFrame(
                {
                    "hour": hours,
                    "meter_id": row.meter_id,
                    "base_kW": base_kW,
                    "heat_kW": heat_kW,
                    "total_kW": total_kW,
                    "price_$perkWh": price_by_hour,
                }
            )
        )

    # all households stacked
    sim_long = pd.concat(sim_rows, ignore_index=True)

    # Household KPIs
    hh_kpis = (
        sim_long.assign(
            kWh=lambda d: d.total_kW * 1.0,
            kWh_heat=lambda d: d.heat_kW * 1.0,
            kWh_base=lambda d: d.base_kW * 1.0,
            cost=lambda d: d.kWh * d["price_$perkWh"],
        )
        .groupby("meter_id", as_index=False)
        .agg(
            kWh_total=("kWh", "sum"),
            kWh_heat=("kWh_heat", "sum"),
            kWh_base=("kWh_base", "sum"),
            bill=("cost", "sum"),
        )
        .rename(columns={"bill": "bill_$"})
    )

    # Feeder KPIs
    feeder_by_hour = sim_long.groupby("hour", as_index=False).agg(
        MW=("total_kW", lambda x: x.sum() / 1000.0)
    )
    t_peak_idx = feeder_by_hour.MW.idxmax()
    t_peak = int(feeder_by_hour.iloc[t_peak_idx].hour)
    peak_MW = float(feeder_by_hour.iloc[t_peak_idx].MW)
    daily_MWh = float(feeder_by_hour.MW.sum() * 1.0)

    feeder_kpis = {
        "peak_MW": peak_MW,
        "t_peak": t_peak,
        "daily_MWh": daily_MWh,
    }

    return sim_long, hh_kpis, feeder_kpis


def _price_series_from_tariff(hours: np.ndarray, tariff: dict) -> np.ndarray:
    prices = np.zeros_like(hours, dtype=float)
    default_price = 0.10
    if not tariff or "periods" not in tariff:
        prices[:] = default_price
        return prices
    for p in tariff["periods"]:
        hrs = set(p.get("hours", []))
        price = float(p.get("price_$perkWh", default_price))
        for i, h in enumerate(hours):
            if int(h) in hrs:
                prices[i] = price
    prices = np.where(prices == 0.0, default_price, prices)
    return prices
