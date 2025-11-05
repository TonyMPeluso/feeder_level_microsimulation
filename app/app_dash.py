# ──────────────────────────────────────────────────────────────────────────────
# File: app/app_dash.py
# Minimal Dash shell for Week‑1 (alternative to Shiny)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px

from src.data_loading import load_archetypes, load_weather
from src.baseload import load_baseload_profiles
from src.tariffs import load_tariff
from src.simulate import simulate_day

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

app = Dash(__name__)
app.layout = html.Div([
    html.H3("Residential Microsim — Week 1 (Dash)"),
    html.Div([
        html.Label("# of meters"),
        dcc.Slider(id="n_meters", min=100, max=5000, step=100, value=1000),
    ], style={"width": "40%"}),
    html.Button("Run simulation", id="run", n_clicks=0),
    dcc.Graph(id="feeder"),
    html.Div(id="summary"),
])

@callback(
    Output("feeder", "figure"),
    Output("summary", "children"),
    Input("run", "n_clicks"),
    Input("n_meters", "value"),
)

def run_sim(n_clicks, n_meters):
    if n_clicks == 0:
        return px.line(pd.DataFrame({"hour":[],"MW":[]}), x="hour", y="MW"), ""

    arch = load_archetypes(str(DATA/"archetypes.csv"))
    weather = load_weather(str(DATA/"weather_winter_design.csv"))
    baseload = load_baseload_profiles(str(DATA/"baseload_profiles.csv"))
    tariff = load_tariff(str(DATA/"tariffs_tou.json"))

    meters = _sample_households(int(n_meters), arch)
    sim_long, hh, fk = simulate_day(meters, weather, baseload, tariff)

    feeder_by_hour = sim_long.groupby("hour", as_index=False).agg(MW=("total_kW", lambda x: x.sum()/1000))
    fig = px.line(feeder_by_hour, x="hour", y="MW", title="Hourly feeder profile (MW)")
    summary = f"Peak: {fk['peak_MW']:.2f} MW at hour {fk['t_peak']} — Daily energy: {fk['daily_MWh']:.1f} MWh"
    return fig, summary


def _sample_households(n: int, arch_df: pd.DataFrame) -> pd.DataFrame:
    probs = (arch_df["Weight"] / arch_df["Weight"].sum()).values
    idx = np.random.choice(arch_df.index, size=n, p=probs)
    sample = arch_df.loc[idx].copy().reset_index(drop=True)
    sample.insert(0, "meter_id", ["M%05d" % i for i in range(n)])
    return sample

if __name__ == "__main__":
    app.run_server(debug=True)


