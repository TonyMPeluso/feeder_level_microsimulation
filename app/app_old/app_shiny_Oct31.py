# ──────────────────────────────────────────────────────────────────────────────
# File: app/app_shiny.py
# Minimal Shiny for Python shell for Week‑1
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import os
import pandas as pd
import numpy as py
from shiny import App, ui, render, reactive
from pathlib import Path

from src.data_loading import load_archetypes, load_weather
from src.baseload import load_baseload_profiles
from src.tariffs import load_tariff
from src.simulate import simulate_day

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_numeric("n_meters", "# of meters", 1000, min=10, max=10000),
        ui.input_select("weather_file", "Weather day", {"weather_winter_design.csv":"Winter design"}),
        ui.input_select("tariff_file", "Tariff", {"tariffs_tou.json":"Simple TOU"}),
        ui.input_action_button("run", "Run simulation"),
    ),
    ui.h2("Residential Microsim — Week 1"),
    ui.layout_columns(
        ui.card(ui.output_plot("feeder_plot")),
        ui.card(ui.output_table("summary_table")),
    ),
    ui.layout_columns(
        ui.card(ui.output_text("peak_text")),
        ui.card(ui.output_table("hh_kpis")),
    ),
)


def server(input, output, session):
    @reactive.event(input.run)
    def _run():
        arch = load_archetypes(str(DATA/"archetypes.csv")) # from data_loading.py
        weather = load_weather(str(DATA/input.weather_file())) # from data_loading.py
        baseload = load_baseload_profiles(str(DATA/"baseload_profiles.csv")) # from baseload.py
        tariff = load_tariff(str((DATA / input.tariff_file())))

        # Sample meters by archetype weights
        meters = _sample_households(input.n_meters(), arch)
        return simulate_day(meters, weather, baseload, tariff) 
        """Returns (sim_long, hh_kpis, feeder_kpis).

        sim_long columns: [hour, meter_id, base_kW, heat_kW, total_kW, price_$perkWh]
        hh_kpis columns: [meter_id, kWh_total, kWh_heat, kWh_base, bill_$]
        feeder_kpis: {"peak_MW": float, "t_peak": int, "daily_MWh": float}
        """

    @output
    @render.plot
    def feeder_plot():
        sim_long, hh, fk = _run()  
        feeder_by_hour = sim_long.groupby("hour", as_index=False).agg(MW=("total_kW", lambda x: x.sum()/1000))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(feeder_by_hour["hour"], feeder_by_hour["MW"], linewidth=2)
        ax.set_xlabel("Hour")
        ax.set_ylabel("Feeder load (MW)")
        ax.set_title("Hourly feeder profile")
        return fig

    @output
    @render.table
    def summary_table():
        sim_long, hh, fk = _run()
        return pd.DataFrame([fk])

    @output
    @render.text
    def peak_text():
        _, __, fk = _run()
        return f"Peak: {fk['peak_MW']:.2f} MW at hour {fk['t_peak']} — Daily energy: {fk['daily_MWh']:.1f} MWh"

    @output
    @render.table
    def hh_kpis():
        _, hh, __ = _run()
        return hh.head(20)


def _sample_households(n: int, arch_df: pd.DataFrame) -> pd.DataFrame:
    rng = pd.Series(arch_df["Weight"].values)
    probs = rng / rng.sum()
    idx = np.random.choice(arch_df.index, size=n, p=probs.values)
    sample = arch_df.loc[idx].copy().reset_index(drop=True)
    sample.insert(0, "meter_id", [f"M{i:05d}" for i in range(n)])
    return sample

app = App(app_ui, server)


