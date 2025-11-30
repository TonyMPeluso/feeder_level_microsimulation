# Feeder-Level Winter Peak Mitigation — Microsimulation

This interactive **Shiny for Python** app models **winter peak electricity demand** at the *feeder* level using a household-based microsimulation.  
It helps utility analysts explore how adoption of **heat pumps**, **thermostat programs**, and **baseload changes** affect feeder load profiles under different customer mixes.

## 🚀 Live Demo (Shiny Cloud)
Try the interactive app here:
👉 https://tonympeluso.shinyapps.io/feeder_microsimulation/

---

## 🌍 Project Overview

Traditional system-wide models often overlook neighborhood-scale effects that drive local capacity upgrades.  
This microsimulation provides a **bottom-up view** of demand — each simulated household has its own archetype, thermal envelope, heating system, and behavior pattern.

Analysts can use this tool to:

- Estimate **feeder peak reductions** from energy-efficiency or demand-response programs
- Examine **variability** across multiple stochastic runs (Monte Carlo ±2σ band)
- Compare **feeder KPIs** (peak kW, peak hour, daily kWh)
- Compare **household KPIs** (energy use, heating share, bill impacts) across archetypes and policy levers

---

## 🧩 Key Features

- Adjustable **household mix** (e.g. apartments, detached, other)
- **Policy levers**:
  - Heat-pump penetration
  - Thermostat setback participation and setpoint deltas
  - Baseload multiplier
- **Monte Carlo** simulation for feeder-level variability (mean ± 2σ band)
- **Median-based KPIs** for robustness to outliers
- Built with [Shiny for Python](https://shiny.posit.co/py/) for a fully interactive experience

---

## 🗂️ Project Structure

```text
feeder_level_microsimulation/
├── app/
│   └── app_shiny.py         # Shiny app entry point
├── src/
│   ├── data_loading.py      # Loads data inputs
│   ├── heating.py           # Generates heat loads, cop, heat transfer coefficient
│   ├── simulate.py          # Core Monte Carlo + microsimulation logic
│   └── tariffs.py           # Loads json tariff file
├── data/
│   ├── archetypes.csv       # Household archetype definitions
│   ├── baseload_profiles.csv# Non-heating load shapes
│   ├── tariffs_tou.json      # Tariff / TOU structure 
│   └── weather_winter_design.csv   # Winter temperature / HDD profiles
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── .gitignore               # Git ignore rules (venv, cache, etc.)
```

🚀 Installation & Running Locally

From the project root (feeder_level_microsimulation/):

1. Create and activate a virtual environment
```
    python3 -m venv .venv
    source .venv/bin/activate        # macOS / Linux
    .venv\Scripts\activate         # Windows (PowerShell / cmd)
```

3. Install dependencies
```
      pip install -r requirements.txt
```

5. Run the Shiny app
```
      python3 -m shiny run --reload app/app_shiny.py
```

Then open your browser at:

http://127.0.0.1:8000

📊 What You See in the App

* Controls panel
* Number of Monte Carlo runs
* Household counts / mix
* Heat-pump share
* Thermostat DR participation and setpoint changes
* Baseload scaling
* Plots
* Histogram of feeder peak load across runs
* Time-series band plot (mean ± 2σ) for feeder load
* KPIs summarizing typical and extreme outcomes

This makes it easy for utility planners to test “What if 30% of customers adopt heat pumps + thermostat DR?” at the feeder level.

🔧 Development Notes

The app is written for Shiny for Python and is compatible with deployment to shinyapps.io via rsconnect-python.

The code is structured so that the microsimulation logic lives in src/, making it reusable in notebooks or batch studies beyond the dashboard.

📫 Contact

Tony Peluso, PhD
Energy Modelling & Grid Analytics
Montreal, QC
