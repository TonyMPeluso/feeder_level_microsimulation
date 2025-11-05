# app/app_shiny.py
from shiny import App, ui, render, reactive, req
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
        ui.h5("Household mix (reweights archetype 'Archetype')"),
        ui.input_slider("share_apartment", "Share: Apartments (%)", 0, 100, 40),
        ui.input_slider("share_detached", "Share: Detached (%)", 0, 100, 40),
        ui.output_text("share_other_text"),
        ui.help_text("‘Other’ updates automatically so that Apartments + Detached + Other = 100%."),

        ui.hr(),
        ui.h5("Variability (Monte Carlo)"),
        ui.input_checkbox("show_band", "Show variability band (multiple runs)", False),
        ui.input_numeric("n_runs", "Number of runs", 20, min=3, max=200),
        ui.input_numeric("mc_seed", "Random seed (optional)", 0, min=0, max=10_000),

        ui.hr(),
        ui.input_action_button("run_btn", "Run simulation"),
    ),

    ui.h2("Feeder-level Winter Peak Mitigation — Microsimulation"),

    # KPIs (Feeder + HH summary) at the top
    ui.card(
        ui.card_header("Feeder KPIs"),
        ui.output_table("feeder_kpis_table"),
    ),

    ui.output_plot("feeder_plot"),

    ui.card(
        ui.card_header("Household KPIs (medians)"),
        ui.output_table("hh_kpis"),
    ),
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _computed_other_pct(apartment_pct: float, detached_pct: float) -> float:
    other = 100.0 - float(apartment_pct) - float(detached_pct)
    return max(0.0, round(other, 2))


def _normalize_mix(apartment_pct: float, detached_pct: float) -> dict:
    other_pct = _computed_other_pct(apartment_pct, detached_pct)
    vec = np.array([apartment_pct, detached_pct, other_pct], dtype=float)
    if vec.sum() <= 0:
        vec = np.array([1.0, 1.0, 1.0])
    vec = vec / vec.sum()
    return {"apartment": vec[0], "detached": vec[1], "other": vec[2]}


def _bucket_from_archetype(a: str) -> str:
    s = str(a).lower()
    if "apt" in s or "apartment" in s or "condo" in s:
        return "apartment"
    if "detached" in s or "house" in s or "sfh" in s:
        return "detached"
    return "other"


def _reweight_archetypes(arch_df: pd.DataFrame, mix: dict) -> pd.Series:
    base_w = arch_df["Weight"] if "Weight" in arch_df.columns else pd.Series(1.0, index=arch_df.index)
    buckets = arch_df["Archetype"].map(_bucket_from_archetype)
    bucket_mult = buckets.map(mix).astype(float)
    w = base_w.astype(float) * bucket_mult
    if w.sum() <= 0:
        w = pd.Series(1.0, index=arch_df.index)
    return (w / w.sum())


def _sample_households(n: int, arch_df: pd.DataFrame, probs: pd.Series) -> pd.DataFrame:
    idx = np.random.choice(arch_df.index, size=int(n), p=probs.values)
    sample = arch_df.loc[idx].copy().reset_index(drop=True)
    sample.insert(0, "meter_id", [f"M{i:05d}" for i in range(int(n))])
    return sample


def _aggregate_feeder(sim_long: pd.DataFrame) -> pd.DataFrame:
    return (
        sim_long.groupby("hour", as_index=False)["total_kW"]
        .sum()
        .rename(columns={"total_kW": "kW"})
    )


def _hh_medians(hh_df: pd.DataFrame) -> dict:
    """Return median HH KPIs for a *single run* across households."""
    return {
        "kWh_total_med": float(hh_df["kWh_total"].median()),
        "kWh_heat_med": float(hh_df["kWh_heat"].median()),
        "kWh_base_med": float(hh_df["kWh_base"].median()),
        "bill_med_$": float(hh_df["bill_$"].median()),
    }


# ---------------------------------------------------------------------
# server
# ---------------------------------------------------------------------
def server(input, output, session):
    DATA = ROOT / "data"

    # computed "Other (%)" text
    @output
    @render.text
    def share_other_text():
        other = _computed_other_pct(input.share_apartment(), input.share_detached())
        return f"Share: Other (%) — {other:.2f}"

    # ---------- single run ----------
    @reactive.calc
    @reactive.event(input.run_btn)
    def _single_run():
        # optional reproducibility
        if input.mc_seed() and int(input.mc_seed()) > 0:
            np.random.seed(int(input.mc_seed()))

        # load data
        arch = load_archetypes(DATA / "archetypes.csv")
        weather = load_weather(DATA / input.weather_file())
        baseload = load_baseload_profiles(DATA / "baseload_profiles.csv")
        tariff = load_tariff(DATA / input.tariff_file())

        # mix controls -> probs
        mix = _normalize_mix(input.share_apartment(), input.share_detached())
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

        # feeder KPI already in fk (peak_MW, t_peak, daily_MWh)
        # HH medians summarized here (for the UI)
        hh_med = _hh_medians(hh)

        return sim_long, hh_med, fk

    # ---------- multiple runs (MC) ----------
    @reactive.calc
    @reactive.event(input.run_btn, input.show_band)
    def _multi_run():
        req(input.show_band())

        base_seed = int(input.mc_seed()) if input.mc_seed() and int(input.mc_seed()) > 0 else None

        # load once
        arch = load_archetypes(DATA / "archetypes.csv")
        weather = load_weather(DATA / input.weather_file())
        baseload = load_baseload_profiles(DATA / "baseload_profiles.csv")
        tariff = load_tariff(DATA / input.tariff_file())

        mix = _normalize_mix(input.share_apartment(), input.share_detached())
        probs = _reweight_archetypes(arch, mix)

        hours = weather["hour"].to_numpy()
        kW_runs = []
        peaks_MW = []
        t_peaks = []
        daily_MWh = []

        # HH median arrays across runs
        hh_kwh_total_meds = []
        hh_kwh_heat_meds = []
        hh_kwh_base_meds = []
        hh_bill_meds = []

        n_runs = int(input.n_runs())
        for r in range(n_runs):
            if base_seed is not None:
                np.random.seed(base_seed + r)
            meters = _sample_households(input.n_meters(), arch, probs)
            w_series = weather.set_index("hour")["T_out_C"]
            sim_long, hh, _ = simulate_day(
                households_df=meters,
                weather_series=w_series,
                baseload_profiles=baseload,
                tariff=tariff,
            )

            # feeder profile for this run
            g = _aggregate_feeder(sim_long).set_index("hour").reindex(hours, fill_value=0.0)
            kW = g["kW"].to_numpy()
            kW_runs.append(kW)

            # feeder KPIs per run
            peak_idx = int(np.argmax(kW))
            peaks_MW.append(float(kW[peak_idx] / 1000.0))
            t_peaks.append(int(hours[peak_idx]))
            daily_MWh.append(float(kW.sum() / 1000.0))  # 1h steps

            # household medians per run
            hh_med = _hh_medians(hh)
            hh_kwh_total_meds.append(hh_med["kWh_total_med"])
            hh_kwh_heat_meds.append(hh_med["kWh_heat_med"])
            hh_kwh_base_meds.append(hh_med["kWh_base_med"])
            hh_bill_meds.append(hh_med["bill_med_$"])

        # hourly statistics for plotting
        arr = np.vstack(kW_runs)  # (n_runs, H)
        df_stats = pd.DataFrame({
            "hour": hours,
            "kW_mean": arr.mean(axis=0),
            "kW_std": arr.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(hours, dtype=float),
        })
        df_stats["kW_m2sd"] = df_stats["kW_mean"] - 2.0 * df_stats["kW_std"]
        df_stats["kW_p2sd"] = df_stats["kW_mean"] + 2.0 * df_stats["kW_std"]

        # KPI bands from distributions across runs
        peaks_MW = np.asarray(peaks_MW)
        peaks_std = float(peaks_MW.std(ddof=1)) if n_runs > 1 else 0.0
        fk = {
            "peak_MW_med": float(np.median(peaks_MW)),
            "peak_MW_m2sd": float(np.median(peaks_MW) - 2.0 * peaks_std),
            "peak_MW_p2sd": float(np.median(peaks_MW) + 2.0 * peaks_std),
            "t_peak_med": int(np.median(t_peaks)),
            "daily_MWh_med": float(np.median(daily_MWh)),
        }

        # HH median-of-medians across runs
        hh_fk = {
            "kWh_total_med": float(np.median(hh_kwh_total_meds)),
            "kWh_heat_med": float(np.median(hh_kwh_heat_meds)),
            "kWh_base_med": float(np.median(hh_kwh_base_meds)),
            "bill_med_$": float(np.median(hh_bill_meds)),
        }

        return df_stats, fk, hh_fk

    # -------------------- outputs --------------------
    @output
    @render.table
    def feeder_kpis_table():
        if input.show_band():
            _, fk, _ = _multi_run()
            df = pd.DataFrame(
                {
                    "Mode": ["Monte Carlo (median across runs)"],
                    "Peak (MW) [median ± 2σ]": [f"{fk['peak_MW_med']:.3f}  [{fk['peak_MW_m2sd']:.3f}, {fk['peak_MW_p2sd']:.3f}]"],
                    "Hour of peak (median)": [int(fk["t_peak_med"])],
                    "Daily energy (MWh, median)": [round(fk["daily_MWh_med"], 3)],
                }
            )
        else:
            _, __, fk = _single_run()
            df = pd.DataFrame(
                {
                    "Mode": ["Single run"],
                    "Peak (MW)": [round(fk["peak_MW"], 3)],
                    "Hour of peak": [int(fk["t_peak"])],
                    "Daily energy (MWh)": [round(fk["daily_MWh"], 3)],
                }
            )
        return df

    @output
    @render.plot
    def feeder_plot():
        import matplotlib.pyplot as plt

        if input.show_band():
            df_stats, _, _ = _multi_run()
            fig, ax = plt.subplots()
            ax.plot(df_stats["hour"], df_stats["kW_mean"] / 1000.0, label="Mean")
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
    def hh_kpis():
        if input.show_band():
            # median of per-run household medians (robust summary)
            _, __, hh_fk = _multi_run()
            df = pd.DataFrame(
                {
                    "kWh (total) — median of medians": [round(hh_fk["kWh_total_med"], 2)],
                    "kWh (heat) — median of medians": [round(hh_fk["kWh_heat_med"], 2)],
                    "kWh (base) — median of medians": [round(hh_fk["kWh_base_med"], 2)],
                    "Bill ($) — median of medians": [round(hh_fk["bill_med_$"], 2)],
                }
            )
            return df
        else:
            _, hh_med, _ = _single_run()
            df = pd.DataFrame(
                {
                    "kWh (total) — median": [round(hh_med["kWh_total_med"], 2)],
                    "kWh (heat) — median": [round(hh_med["kWh_heat_med"], 2)],
                    "kWh (base) — median": [round(hh_med["kWh_base_med"], 2)],
                    "Bill ($) — median": [round(hh_med["bill_med_$"], 2)],
                }
            )
            return df


app = App(app_ui, server)