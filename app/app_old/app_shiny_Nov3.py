# app/app_shiny.py

from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- make project root importable ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loading import (
    load_archetypes,
    load_weather,
    load_baseload_profiles,
    load_tariff,
)
from src.simulate import simulate_day


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_numeric("n_meters", "Number of meters", 200, min=10, max=5000),
        ui.input_slider(
            "bl_mult",
            "Baseload multiplier",
            min=0.5,
            max=1.5,
            value=1.0,
            step=0.05,
        ),
        ui.input_numeric("t_day", "Day setpoint (°C)", 21),
        ui.input_numeric("t_night", "Night setpoint (°C)", 19),
        ui.input_select(
            "weather_file",
            "Weather",
            {
                "weather_winter_design.csv": "Winter design day",
            },
        ),
        ui.input_select(
            "tariff_file",
            "Tariff",
            {
                "tariffs_tou.json": "Time-of-use",
            },
        ),
        ui.input_action_button("run_btn", "Run simulation"),
    ),
    ui.h2("Winter peak mitigation — microsim draft"),
    ui.output_plot("feeder_plot"),
    # KPI row
    ui.layout_columns(
        ui.card(
            ui.card_header("Peak (MW)"),
            ui.output_text("kpi_peak"),
        ),
        ui.card(
            ui.card_header("Daily energy (MWh)"),
            ui.output_text("kpi_energy"),
        ),
        ui.card(
            ui.card_header("# of meters"),
            ui.output_text("kpi_meters"),
        ),
    ),
    ui.download_button("dl_sim", "Download hourly sim (CSV)"),
    ui.h3("Household KPIs"),
    ui.output_table("hh_kpis"),
)


# ---------------------------------------------------------------------
# server
# ---------------------------------------------------------------------
def server(input, output, session):
    DATA = ROOT / "data"

    # 1) build the synthetic customer set from archetypes
    @reactive.calc
    def households_df() -> pd.DataFrame:
        arch = load_archetypes(DATA / "archetypes.csv")

        # sample households according to the Weight column
        n = int(input.n_meters())
        rng = pd.Series(arch["Weight"].values, index=arch.index)
        probs = rng / rng.sum()
        idx = np.random.choice(arch.index, size=n, p=probs.values)
        hh = arch.loc[idx].copy().reset_index(drop=True)

        # meter ids
        hh.insert(0, "meter_id", [f"M{i:05d}" for i in range(n)])

        # apply UI-driven overrides
        hh["Baseload_Mult"] = float(input.bl_mult())
        hh["Setpoint_Day_C"] = float(input.t_day())
        hh["Setpoint_Night_C"] = float(input.t_night())

        return hh

    # 2) run the simulation whenever inputs change
    @reactive.calc
    def sim_result():
        # depend on the button so we don't re-run on every tiny change
        input.run_btn()

        hh = households_df()
        weather = load_weather(DATA / input.weather_file())
        baseload = load_baseload_profiles(DATA / "baseload_profiles.csv")
        tariff = load_tariff(DATA / input.tariff_file())

        # weather: expect col hour, T_out_C
        w_series = weather.set_index("hour")["T_out_C"]

        sim_long, hh_kpis, feeder_kpis = simulate_day(
            households_df=hh,
            weather_series=w_series,
            baseload_profiles=baseload,
            tariff=tariff,
        )
        return sim_long, hh_kpis, feeder_kpis

    # -----------------------------------------------------------------
    # outputs
    # -----------------------------------------------------------------
    @output
    @render.plot
    def feeder_plot():
        import matplotlib.pyplot as plt

        sim_long, _, fk = sim_result()
        # aggregate to hour
        g = sim_long.groupby("hour", as_index=False)["total_kW"].sum()

        fig, ax = plt.subplots()
        ax.plot(g["hour"], g["total_kW"] / 1000.0, label="Feeder load")
        # vertical line at peak
        ax.axvline(fk["t_peak"], linestyle="--", label="Peak hour")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Feeder load (MW)")
        ax.set_title("Feeder load profile")
        ax.legend()
        return fig

    @output
    @render.text
    def kpi_peak():
        _, __, fk = sim_result()
        return f"{fk['peak_MW']:.2f}"

    @output
    @render.text
    def kpi_energy():
        _, __, fk = sim_result()
        return f"{fk['daily_MWh']:.1f}"

    @output
    @render.text
    def kpi_meters():
        hh = households_df()
        return str(len(hh))

    @output
    @render.table
    def hh_kpis():
        _, hh, _ = sim_result()
        # show the first ~200 rows to keep UI snappy
        return hh.head(200)

    @output
    @render.download(filename="sim_long.csv")
    def dl_sim():
        sim_long, _, _ = sim_result()
        return sim_long.to_csv(index=False)


app = App(app_ui, server)
