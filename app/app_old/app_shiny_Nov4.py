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

        ui.input_select(
            "weather_file", "Weather",
            {"weather_winter_design.csv": "Winter design day"}
        ),
        ui.input_select(
            "tariff_file", "Tariff",
            {"tariffs_tou.json": "Time-of-use"}
        ),

        ui.hr(),
        ui.h5("Household mix (reweights archetype 'Archetype' values)"),
        ui.input_slider("share_apartment", "Share: Apartments (%)", 0, 100, 40),
        ui.input_slider("share_detached", "Share: Detached (%)", 0, 100, 40),
        ui.input_slider("share_other", "Share: Other (%)", 0, 100, 20),
        ui.help_text("These three are normalized to 100% when sampling."),

        ui.hr(),
        ui.h5("Variability (Monte Carlo)"),
        ui.input_checkbox("show_band", "Show variability band (multiple runs)", False),
        ui.input_numeric("n_runs", "Number of runs", 20, min=3, max=200),
        ui.input_numeric("mc_seed", "Random seed (optional)", 0, min=0, max=10_000),

        ui.hr(),
        ui.input_action_button("run_btn", "Run simulation"),
    ),
    ui.h2("Winter peak mitigation — microsim draft"),
    ui.output_plot("feeder_plot"),
    ui.output_table("summary_table"),
    ui.output_text("peak_text"),
    ui.output_table("hh_kpis"),
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _normalize_mix(apartment_pct: float, detached_pct: float, other_pct: float) -> dict:
    vec = np.array([apartment_pct, detached_pct, other_pct], dtype=float)
    if vec.sum() <= 0:
        vec = np.array([1.0, 1.0, 1.0])
    vec = vec / vec.sum()
    return {"apartment": vec[0], "detached": vec[1], "other": vec[2]}


def _bucket_from_archetype(a: str) -> str:
    """Map freeform/CSV 'Archetype' names to mix buckets. Adjust as needed."""
    s = str(a).lower()
    if "apt" in s or "apartment" in s or "condo" in s:
        return "apartment"
    if "detached" in s or "house" in s or "sfh" in s:
        return "detached"
    return "other"


def _reweight_archetypes(arch_df: pd.DataFrame, mix: dict) -> pd.Series:
    """
    Return a probability per row for sampling, based on:
    - row Weight (if present) AND
    - user mix for apartment/detached/other
    """
    base_w = arch_df["Weight"] if "Weight" in arch_df.columns else pd.Series(1.0, index=arch_df.index)
    buckets = arch_df["Archetype"].map(_bucket_from_archetype)
    bucket_mult = buckets.map(mix).astype(float)
    # multiply row weight by bucket multiplier
    w = base_w.astype(float) * bucket_mult
    # guard against all-zero
    if w.sum() <= 0:
        w = pd.Series(1.0, index=arch_df.index)
    return (w / w.sum())


def _sample_households(n: int, arch_df: pd.DataFrame, probs: pd.Series) -> pd.DataFrame:
    idx = np.random.choice(arch_df.index, size=int(n), p=probs.values)
    sample = arch_df.loc[idx].copy().reset_index(drop=True)
    # make meter ids
    sample.insert(0, "meter_id", [f"M{i:05d}" for i in range(int(n))])
    return sample


def _aggregate_feeder(sim_long: pd.DataFrame) -> pd.DataFrame:
    return sim_long.groupby("hour", as_index=False)["total_kW"].sum().rename(columns={"total_kW": "kW"})


# ---------------------------------------------------------------------
# server
# ---------------------------------------------------------------------
def server(input, output, session):
    DATA = ROOT / "data"

    @reactive.calc
    @reactive.event(input.run_btn)  # ← add this
    def _single_run():
        # gate by button
        # reactive.require(input.run_btn() > 0)

        # fix seed for reproducibility within a "run" if user supplied one
        if input.mc_seed() and int(input.mc_seed()) > 0:
            np.random.seed(int(input.mc_seed()))

        # load data
        arch = load_archetypes(DATA / "archetypes.csv")
        weather = load_weather(DATA / input.weather_file())
        baseload = load_baseload_profiles(DATA / "baseload_profiles.csv")
        tariff = load_tariff(DATA / input.tariff_file())

        # mix controls -> probs
        mix = _normalize_mix(input.share_apartment(), input.share_detached(), input.share_other())
        probs = _reweight_archetypes(arch, mix)

        # sample households
        meters = _sample_households(input.n_meters(), arch, probs)

        # simulate
        w_series = weather.set_index("hour")["T_out_C"]
        sim_long, hh, fk = simulate_day(
            households_df=meters,
            weather_series=w_series,
            baseload_profiles=baseload,
            tariff=tariff,
        )
        return sim_long, hh, fk

    @reactive.calc
    @reactive.event(input.run_btn, input.show_band)  # retrigger when toggled
    def _multi_run():
        # gate by button AND option
        # reactive.require(input.run_btn() > 0)
        # reactive.require(input.show_band())

        # Seed pattern: different draws across runs but reproducible across clicks if mc_seed set
        base_seed = int(input.mc_seed()) if input.mc_seed() and int(input.mc_seed()) > 0 else None

        # load once
        arch = load_archetypes(DATA / "archetypes.csv")
        weather = load_weather(DATA / input.weather_file())
        baseload = load_baseload_profiles(DATA / "baseload_profiles.csv")
        tariff = load_tariff(DATA / input.tariff_file())

        mix = _normalize_mix(input.share_apartment(), input.share_detached(), input.share_other())
        probs = _reweight_archetypes(arch, mix)

        hours = weather["hour"].to_numpy()
        kW_runs = []

        n_runs = int(input.n_runs())
        for r in range(n_runs):
            if base_seed is not None:
                np.random.seed(base_seed + r)
            meters = _sample_households(input.n_meters(), arch, probs)
            w_series = weather.set_index("hour")["T_out_C"]
            sim_long, _, _ = simulate_day(
                households_df=meters,
                weather_series=w_series,
                baseload_profiles=baseload,
                tariff=tariff,
            )
            g = _aggregate_feeder(sim_long)
            # ensure alignment
            g = g.set_index("hour").reindex(hours, fill_value=0.0)
            kW_runs.append(g["kW"].to_numpy())

        arr = np.vstack(kW_runs)  # shape (n_runs, H)
        stats = {
            "hour": hours,
            "kW_mean": arr.mean(axis=0),
            "kW_std": arr.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(hours, dtype=float),
        }
        df_stats = pd.DataFrame(stats)
        df_stats["kW_p10"] = df_stats["kW_mean"] - 1.2816 * df_stats["kW_std"]
        df_stats["kW_p90"] = df_stats["kW_mean"] + 1.2816 * df_stats["kW_std"]
        df_stats["kW_m2sd"] = df_stats["kW_mean"] - 2.0 * df_stats["kW_std"]
        df_stats["kW_p2sd"] = df_stats["kW_mean"] + 2.0 * df_stats["kW_std"]

        # peak statistics
        peak_idx = int(df_stats["kW_mean"].idxmax())
        peak_hour = int(df_stats.iloc[peak_idx]["hour"])
        peak_MW = float(df_stats.iloc[peak_idx]["kW"] / 1000.0) if "kW" in df_stats.columns else float(df_stats.iloc[peak_idx]["kW_mean"] / 1000.0)
        daily_MWh = float(df_stats["kW_mean"].sum() / 1000.0)  # 1h steps

        fk = {"peak_MW": peak_MW, "t_peak": peak_hour, "daily_MWh": daily_MWh}
        return df_stats, fk

    # -------------------- outputs --------------------
    @output
    @render.plot
    def feeder_plot():
        import matplotlib.pyplot as plt

        if input.show_band():
            df_stats, _ = _multi_run()
            fig, ax = plt.subplots()
            ax.plot(df_stats["hour"], df_stats["kW_mean"] / 1000.0, label="Mean")
            # 2σ band (shaded)
            ax.fill_between(
                df_stats["hour"],
                (df_stats["kW_m2sd"] / 1000.0).clip(lower=0),
                (df_stats["kW_p2sd"] / 1000.0),
                alpha=0.2,
                label="±2σ band",
            )
            ax.set_xlabel("Hour")
            ax.set_ylabel("Feeder load (MW)")
            ax.set_title("Feeder load profile — mean ± 2σ")
            ax.legend()
            return fig
        else:
            sim_long, _, _ = _single_run()
            g = _aggregate_feeder(sim_long)
            fig, ax = plt.subplots()
            ax.plot(g["hour"], g["kW"] / 1000.0)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Feeder load (MW)")
            ax.set_title("Feeder load profile")
            return fig

    @output
    @render.table
    def summary_table():
        if input.show_band():
            df_stats, _ = _multi_run()
            # Show hourly mean and ±2σ in MW for readability
            out = pd.DataFrame({
                "hour": df_stats["hour"],
                "MW_mean": df_stats["kW_mean"] / 1000.0,
                "MW_minus_2sd": (df_stats["kW_m2sd"] / 1000.0).clip(lower=0),
                "MW_plus_2sd": df_stats["kW_p2sd"] / 1000.0,
            })
            return out
        else:
            _, hh, _ = _single_run()
            return hh

    @output
    @render.text
    def peak_text():
        if input.show_band():
            _, fk = _multi_run()
            return (
                f"[MC] Mean peak ≈ {fk['peak_MW']:.3f} MW at hour {fk['t_peak']} — "
                f"Daily energy (mean): {fk['daily_MWh']:.3f} MWh"
            )
        else:
            _, __, fk = _single_run()
            return (
                f"Peak: {fk['peak_MW']:.3f} MW at hour {fk['t_peak']} — "
                f"Daily energy: {fk['daily_MWh']:.3f} MWh"
            )

    @output
    @render.table
    def hh_kpis():
        if input.show_band():
            # Not meaningful across MC runs (households change each run)
            return pd.DataFrame({"Note": ["Household KPIs hidden in MC mode (mix varies each run)."]})
        else:
            _, hh, _ = _single_run()
            return hh


app = App(app_ui, server)
