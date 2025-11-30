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
    # 1) Sidebar MUST be first
    ui.sidebar(
        ui.h4("Scenario inputs"),
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
        ui.h5("Household Mix (Archetype Weighting)"),
        ui.input_slider("share_apartment", "Apartments (%)", 0, 100, 40),
        ui.input_slider("share_detached", "Detached (%)", 0, 100, 40),
        ui.output_text("share_other_text"),

        ui.hr(),
        ui.h5("Policy Levers"),
        ui.input_slider("hp_share", "Target heat pump share (%)", 0, 100, 50),
        ui.input_slider("setback_participation", "Thermostat program participation (%)", 0, 100, 60),
        ui.input_slider("delta_day", "Day setpoint change (°C)", -3, 3, -1),
        ui.input_slider("delta_night", "Night setpoint change (°C)", -3, 3, -2),
        ui.input_slider("baseload_mult", "Baseload multiplier (×)", 0.5, 1.5, 1.0, step=0.05),

        ui.hr(),
        ui.h5("Variability (Monte Carlo)"),
        ui.input_checkbox("show_band", "Show variability band (multiple runs)", False),
        ui.input_numeric("n_runs", "Number of runs", 20, min=3, max=200),
        ui.input_numeric("mc_seed", "Random seed (optional)", 0, min=0, max=10_000),

        ui.hr(),
        ui.h5("Transformer / Feeder Limits"),
        ui.input_numeric(
            "feeder_rating_mw",
            "Feeder / transformer rating (MW)",
            5.0,
            min=0.1,
            max=100.0,
        ),

        ui.hr(),
        ui.input_action_button("run_btn", "Run simulation", class_="btn-primary"),
    ),

    # 2) Global CSS injected into <head>
    ui.head_content(
        ui.tags.style(
            """
            body {
                font-size: 16px;
            }
            table {
                font-size: 16px;
            }
            .kpi-table {
                font-size: 16px;
                border-collapse: collapse;
                width: 100%;
            }
            .kpi-table th, .kpi-table td {
                padding: 4px 8px;
            }
            /* Right-align the last column (values) */
            .kpi-table td:last-child {
                text-align: right;
            }
            """
        )
    ),

    # 3) Main page content
    ui.h2("Feeder-level Winter Peak Mitigation — Microsimulation"),
    ui.output_plot("feeder_plot"),
    ui.layout_columns(
        ui.card(
            ui.card_header("Feeder KPIs"),
            ui.output_ui("feeder_kpi_table"),
        ),
        ui.card(
            ui.card_header("Household KPIs"),
            ui.output_ui("household_kpi_table"),
        ),
        width=1/2,
    ),
    ui.h4("Overload risk (Monte Carlo mode)"),
    ui.output_table("overload_kpis"),
    ui.download_button("download_overload_events", "Download overload events (CSV)"),
    ui.output_text("note_text"),
)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _normalize_mix(apartment_pct: float, detached_pct: float) -> tuple[dict, float]:
    other_pct = max(0.0, 100.0 - float(apartment_pct) - float(detached_pct))
    vec = np.array([apartment_pct, detached_pct, other_pct], dtype=float)
    s = vec.sum()
    if s <= 0:
        vec = np.array([1.0, 1.0, 1.0])
        s = 3.0
    vec = vec / s
    return {"apartment": vec[0], "detached": vec[1], "other": vec[2]}, other_pct


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


def _apply_policies(
    meters: pd.DataFrame,
    hp_share_pct: float,
    delta_day: float,
    delta_night: float,
    setback_participation_pct: float,
    baseload_mult: float,
) -> pd.DataFrame:
    out = meters.copy()

    # Heat pump conversion
    hs = out["HeatSystem"].astype(str).str.lower()
    n = len(out)
    target_n_hp = int(round((hp_share_pct / 100.0) * n))
    current_n_hp = int((hs == "heat_pump").sum())
    to_convert = max(0, target_n_hp - current_n_hp)
    if to_convert > 0:
        non_hp_idx = out.index[hs != "heat_pump"].to_numpy()
        if len(non_hp_idx) > 0:
            chosen = np.random.choice(
                non_hp_idx,
                size=min(to_convert, len(non_hp_idx)),
                replace=False,
            )
            out.loc[chosen, "HeatSystem"] = "heat_pump"

    # Thermostat setback program
    p = max(0.0, min(100.0, float(setback_participation_pct))) / 100.0
    if p > 0 and (delta_day != 0 or delta_night != 0):
        mask = np.random.rand(len(out)) < p
        for col, d in (("Setpoint_Day_C", delta_day), ("Setpoint_Night_C", delta_night)):
            if col in out.columns and d != 0:
                out.loc[mask, col] = (
                    out.loc[mask, col].astype(float) + float(d)
                ).clip(15.0, 24.0)

    # Baseload multiplier
    out["Baseload_Mult"] = float(baseload_mult)
    return out


def _aggregate_feeder(sim_long: pd.DataFrame) -> pd.DataFrame:
    return (
        sim_long.groupby("hour", as_index=False)["total_kW"]
        .sum()
        .rename(columns={"total_kW": "kW"})
    )


# ---------------------------------------------------------------------
# server
# ---------------------------------------------------------------------
def server(input, output, session):
    DATA = ROOT / "data"

    @output
    @render.text
    def share_other_text():
        mix, other_pct = _normalize_mix(input.share_apartment(), input.share_detached())
        return f"Other (%) auto: {other_pct:.1f} (normalized internally)"

    # ---- single run ----
    @reactive.calc
    @reactive.event(input.run_btn)
    def _single_run():
        # Optional fixed seed for reproducibility in single-run mode
        if input.mc_seed() and int(input.mc_seed()) > 0:
            np.random.seed(int(input.mc_seed()))

        arch = load_archetypes(DATA / "archetypes.csv")
        weather = load_weather(DATA / input.weather_file())
        baseload = load_baseload_profiles(DATA / "baseload_profiles.csv")
        tariff = load_tariff(DATA / input.tariff_file())

        mix, _ = _normalize_mix(input.share_apartment(), input.share_detached())
        probs = _reweight_archetypes(arch, mix)
        meters = _sample_households(input.n_meters(), arch, probs)
        meters = _apply_policies(
            meters,
            hp_share_pct=input.hp_share(),
            delta_day=input.delta_day(),
            delta_night=input.delta_night(),
            setback_participation_pct=input.setback_participation(),
            baseload_mult=input.baseload_mult(),
        )

        w_series = weather.set_index("hour")["T_out_C"]
        sim_long, hh, fk = simulate_day(meters, w_series, baseload, tariff)
        return sim_long, hh, fk

    # ---- multi-run (Monte Carlo) ----
    @reactive.calc
    @reactive.event(input.run_btn, input.show_band)
    def _multi_run():
        req(input.show_band())

        arch = load_archetypes(DATA / "archetypes.csv")
        weather = load_weather(DATA / input.weather_file())
        baseload = load_baseload_profiles(DATA / "baseload_profiles.csv")
        tariff = load_tariff(DATA / input.tariff_file())

        mix, _ = _normalize_mix(input.share_apartment(), input.share_detached())
        probs = _reweight_archetypes(arch, mix)

        hours = weather["hour"].to_numpy()
        n_runs = int(input.n_runs())
        base_seed = int(input.mc_seed()) if input.mc_seed() and int(input.mc_seed()) > 0 else None

        # Feeder rating in kW
        rating_mw = float(input.feeder_rating_mw())
        rating_kw = rating_mw * 1000.0

        # To accumulate
        kW_runs = []
        peak_MW_runs = []
        peak_hr_runs = []
        daily_MWh_runs = []

        hh_kwh_total_meds = []
        hh_kwh_heat_meds = []
        hh_kwh_base_meds = []
        hh_bill_meds = []

        # Overload-related
        overload_run_has = []
        overload_hours_per_run = []
        max_overload_kw = 0.0
        event_rows = []

        for r in range(n_runs):
            if base_seed is not None:
                np.random.seed(base_seed + r)

            meters = _sample_households(input.n_meters(), arch, probs)
            meters = _apply_policies(
                meters,
                hp_share_pct=input.hp_share(),
                delta_day=input.delta_day(),
                delta_night=input.delta_night(),
                setback_participation_pct=input.setback_participation(),
                baseload_mult=input.baseload_mult(),
            )

            w_series = weather.set_index("hour")["T_out_C"]
            sim_long, hh, _ = simulate_day(meters, w_series, baseload, tariff)

            g = _aggregate_feeder(sim_long).set_index("hour").reindex(hours, fill_value=0.0)
            k = g["kW"].to_numpy()
            kW_runs.append(k)

            # Basic feeder KPIs
            peak_idx = int(np.argmax(k))
            peak_hr = int(hours[peak_idx])
            peak_kw = float(k[peak_idx])
            peak_MW_runs.append(peak_kw / 1000.0)
            peak_hr_runs.append(peak_hr)
            daily_MWh_runs.append(float(k.sum() / 1000.0))

            # Household medians
            hh_kwh_total_meds.append(float(hh["kWh_total"].median()))
            hh_kwh_heat_meds.append(float(hh["kWh_heat"].median()))
            hh_kwh_base_meds.append(float(hh["kWh_base"].median()))
            hh_bill_meds.append(float(hh["bill_$"].median()))

            # Overload events (relative to feeder rating)
            overload_mask = k > rating_kw
            n_over = int(overload_mask.sum())
            has_over = n_over > 0
            overload_run_has.append(has_over)
            overload_hours_per_run.append(n_over)

            if has_over:
                overload_kw = k[overload_mask] - rating_kw
                if overload_kw.size > 0:
                    max_overload_kw = max(max_overload_kw, float(overload_kw.max()))

                # Build rows for event log
                for h, kw_val, ov_kw in zip(hours[overload_mask], k[overload_mask], overload_kw):
                    event_rows.append(
                        {
                            "run_id": r + 1,
                            "hour": int(h),
                            "load_MW": float(kw_val / 1000.0),
                            "overload_MW": float(ov_kw / 1000.0),
                        }
                    )

        # Hourly mean/std band
        arr = np.vstack(kW_runs)
        df_stats = pd.DataFrame(
            {
                "hour": hours,
                "kW_mean": arr.mean(axis=0),
                "kW_std": arr.std(axis=0, ddof=1) if n_runs > 1 else np.zeros_like(hours, dtype=float),
            }
        )
        df_stats["kW_m2sd"] = df_stats["kW_mean"] - 2 * df_stats["kW_std"]
        df_stats["kW_p2sd"] = df_stats["kW_mean"] + 2 * df_stats["kW_std"]

        # Median feeder KPIs across runs
        fk_median = {
            "peak_MW": float(np.median(peak_MW_runs)),
            "t_peak": int(round(float(np.median(peak_hr_runs)))),
            "daily_MWh": float(np.median(daily_MWh_runs)),
        }

        # Median household KPIs across runs
        hh_medians = {
            "kWh_total_med": float(np.median(hh_kwh_total_meds)),
            "kWh_heat_med": float(np.median(hh_kwh_heat_meds)),
            "kWh_base_med": float(np.median(hh_kwh_base_meds)),
            "bill$_med": float(np.median(hh_bill_meds)),
        }

        # Overload risk KPIs
        if len(overload_run_has) > 0:
            prob_overload = float(np.mean(overload_run_has))  # fraction of runs with any overload
            mean_hours_over = float(np.mean(overload_hours_per_run))
        else:
            prob_overload = 0.0
            mean_hours_over = 0.0

        overload_kpis = pd.DataFrame(
            {
                "metric": [
                    "Probability of any overload (per run)",
                    "Probability of any overload (%)",
                    "Mean overload hours per run",
                    "Maximum overload (MW) across all runs",
                    "Feeder rating (MW)",
                ],
                "value": [
                    prob_overload,
                    prob_overload * 100.0,
                    mean_hours_over,
                    float(max_overload_kw / 1000.0),
                    rating_mw,
                ],
            }
        )

        # Overload event log
        if event_rows:
            events_df = pd.DataFrame(event_rows).sort_values(["run_id", "hour"])
        else:
            events_df = pd.DataFrame(columns=["run_id", "hour", "load_MW", "overload_MW"])

        return df_stats, fk_median, hh_medians, overload_kpis, events_df

    # -------------------- outputs --------------------
    @output
    @render.plot
    def feeder_plot():
        import matplotlib.pyplot as plt

        if input.show_band():
            df_stats, _, _, _, _ = _multi_run()
            fig, ax = plt.subplots()
            ax.plot(df_stats["hour"], df_stats["kW_mean"] / 1000.0, label="Mean")
            ax.fill_between(
                df_stats["hour"],
                (df_stats["kW_m2sd"] / 1000.0).clip(lower=0),
                df_stats["kW_p2sd"] / 1000.0,
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
    @render.ui
    def feeder_kpi_table():
        # Build the same data as before
        if input.show_band():
            _, fk, _, _, _ = _multi_run()
            data = {
                "metric": [
                    "Peak (MW) — median across runs",
                    "Peak hour — median across runs",
                    "Daily MWh — median across runs",
                ],
                "value": [fk["peak_MW"], fk["t_peak"], fk["daily_MWh"]],
            }
        else:
            _, __, fk = _single_run()
            data = {
                "metric": ["Peak (MW)", "Peak hour", "Daily MWh"],
                "value": [fk["peak_MW"], fk["t_peak"], fk["daily_MWh"]],
            }

        df = pd.DataFrame(data)

        def fmt(v):
            if isinstance(v, (int, float)):
                return f"{v:0.2f}"
            return str(v)

        df["value"] = df["value"].apply(fmt)

        html = df.to_html(
            index=False,               # <-- removes the row numbers
            classes="kpi-table table table-sm",
            border=0,
        )
        return ui.HTML(html)

    @output
    @render.ui
    def household_kpi_table():
        if input.show_band():
            _, __, hhmed, _, _ = _multi_run()
            data = {
                "metric": [
                    "Per-household kWh — median (across runs)",
                    "Per-household heating kWh — median (across runs)",
                    "Per-household baseload kWh — median (across runs)",
                    "Per-household bill ($) — median (across runs)",
                ],
                "value": [
                    hhmed["kWh_total_med"],
                    hhmed["kWh_heat_med"],
                    hhmed["kWh_base_med"],
                    hhmed["bill$_med"],
                ],
            }
        else:
            _, hh, _ = _single_run()
            data = {
                "metric": [
                    "Per-household kWh — median",
                    "Per-household heating kWh — median",
                    "Per-household baseload kWh — median",
                    "Per-household bill ($) — median",
                ],
                "value": [
                    float(hh["kWh_total"].median()),
                    float(hh["kWh_heat"].median()),
                    float(hh["kWh_base"].median()),
                    float(hh["bill_$"].median()),
                ],
            }

        df = pd.DataFrame(data)

        def fmt(v):
            if isinstance(v, (int, float)):
                return f"{v:0.2f}"
            return str(v)

        df["value"] = df["value"].apply(fmt)

        html = df.to_html(
            index=False,               # <-- removes the row numbers
            classes="kpi-table table table-sm",
            border=0,
        )
        return ui.HTML(html)

    @output
    @render.table
    def overload_kpis():
        # Only meaningful in Monte Carlo mode
        if not input.show_band():
            return pd.DataFrame(
                {
                    "metric": ["Overload risk metrics are only computed in Monte Carlo mode."],
                    "value": [""],
                }
            )

        _, __, ___, kdf, ____ = _multi_run()

        def fmt(v):
            if isinstance(v, (int, float)):
                return f"{v:>8.3f}"
            return str(v)

        kdf = kdf.copy()
        kdf["value"] = kdf["value"].apply(fmt)
        return kdf


    @output
    @render.download(filename="overload_events.csv")
    def download_overload_events():
        # In single-run mode, just provide an empty log with headers.
        if not input.show_band():
            empty = pd.DataFrame(columns=["run_id", "hour", "load_MW", "overload_MW"])
            return empty.to_csv(index=False).encode("utf-8")

        _, __, ___, ____, events_df = _multi_run()
        return events_df.to_csv(index=False).encode("utf-8")

    @output
    @render.text
    def note_text():
        return (
            "Note: In Monte Carlo mode, feeder KPIs are medians across runs; "
            "household KPIs are medians of each run's household medians. "
            "Overload risk metrics and the overload event log are also computed "
            "only in Monte Carlo mode, relative to the Feeder / transformer rating (MW)."
        )


app = App(app_ui, server)