# Feeder-Level Winter Peak Mitigation — Microsimulation

This interactive **Python Shiny** app models **winter peak electricity demand** at the *feeder* level using a household-based microsimulation.  
It helps utility analysts explore how adoption of **heat pumps**, **thermostat programs**, and **baseload changes** affect feeder load profiles under different customer mixes.

---

## 🌍 Project Overview

Traditional system-wide models often overlook neighborhood-scale effects that drive local capacity upgrades.  
This microsimulation provides a **bottom-up view** of demand — each simulated household has its own archetype, thermal envelope, heating system, and behavior pattern.

Analysts can use this tool to:
- Estimate **feeder peak reductions** from energy-efficiency or demand-response programs.  
- Examine **variability** across multiple stochastic runs (Monte Carlo ±2σ band).  
- Compare **median feeder KPIs** (peak MW, peak hour, daily MWh) and **median household KPIs** (energy use, heating share, bill).  

---

## 🧩 Features

- Adjustable **household mix** (apartments, detached, other)
- **Policy levers**:
  - Heat-pump share
  - Thermostat setback participation and setpoint deltas
  - Baseload multiplier  
- **Monte Carlo** simulation for feeder-level variability (mean ± 2σ band)
- **Median-based KPIs** for feeder and households
- Built with [**Shiny for Python**](https://shiny.posit.co/py/) — fully interactive

---

## 🗂️ Folder Structure

```
winter_peak_mitigation/
├── app/ # Shiny app (entry point: app_shiny.py)
├── src/ # Simulation core modules
├── data/ # Input datasets (archetypes, baseloads, tariffs, weather)
├── requirements.txt # Python dependencies
└── README.md # This file
```

---

## Running the App

1. Create a virtual environment:
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

2. Install dependencies:
pip install -r requirements.txt

3. Launch the Shiny app:
shiny run --reload app/app_shiny.py

4. Open your browser at 
http://127.0.0.1:8000
