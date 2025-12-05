# Feeder-Level Winter Peak Mitigation — Microsimulation #

Interactive Python Shiny microsimulation for analyzing feeder-level peak electricity demand under winter conditions.
Models household heating systems, baseload diversity, and behavioral variability to evaluate peak mitigation strategies and distribution-level risks.

🚀 ##Live Demo##

👉 https://tonympeluso.shinyapps.io/feeder_microsimulation/

🌍 ## Overview

Traditional system-wide models often miss neighborhood-scale dynamics that drive localized feeder upgrades, transformer overloads, and winter peak risk.

This project implements a bottom-up household microsimulation, where each dwelling has:
* its own thermal envelope
* heating system (resistance, furnace, heat pump)
* baseload profile
* thermostat behavior
* stochastic variations

The result is an interactive explorer for utility planners, engineers, and policy analysts evaluating:
* heat-pump adoption scenarios
* thermostat setback / DR participation
* impacts on feeder peaks
* variability across Monte Carlo runs
* household-level energy + bill effects

🧩 ## Key Features
Household Mix & Archetypes
* Adjustable weighting of apartments / detached / other
* Flexible archetype definitions

# Policy Levers
* Heat-pump penetration targets
* Thermostat setback participation
* Day/night setpoint deltas
* Baseload multiplier

# Monte Carlo Engine
* Multiple stochastic simulation runs
*  Mean ± 2σ variability band
* Median KPIs to avoid outlier distortion

# KPIs
Feeder-level
* Peak MW
* Peak hour
* Daily MWh
* Overload probability (vs feeder rating)

Household-level
* Total kWh
* Heating share
* Baseload share
* Median energy bill

Technology
* Built with Shiny for Python
* Modular backend in src/ for independent use in notebooks

📊 ## Screenshots / Outputs (placeholders for now)
* You can add these later:
* Feeder load curve (single run)
* Monte Carlo mean ± 2σ band
* Feeder KPIs card
* Household KPIs card
* Histogram of peak loads
*Overload event log

I can generate sample graphics if you want.

🗂️ ## Project Structure
```
feeder_level_microsimulation/
├── app/
│   └── app_shiny.py           # Shiny UI & server
├── src/
│   ├── data_loading.py        # Data ingestion utilities
│   ├── heating.py             # Thermal + heating model
│   ├── simulate.py            # Microsimulation + Monte Carlo engine
│   └── tariffs.py             # Tariff structure loader
├── data/
│   ├── archetypes.csv         # Household archetypes
│   ├── baseload_profiles.csv  # Non-heating profiles
│   ├── tariffs_tou.json       # Tariff structure
│   └── weather_winter_design.csv  # Winter design temperatures
├── requirements.txt           
├── README.md
└── .gitignore
```

⚙️ ## Installation & Running Locally

Create virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# or
.venv\Scripts\activate           # Windows
```

Install dependencies:
```
pip install -r requirements.txt
```

Run locally:
```
python3 -m shiny run --reload app/app_shiny.py
```

Then visit:

http://127.0.0.1:8000

🧠 ## Modelling Approach
# Thermal Model
* Heat loss via UA value
* Temperature-dependent heating load
* Heat-pump COP curve
* Night/day setpoint control

# Baseload Model
* Archetype-specific profiles
* Multipliers for policy scenarios

# Stochastic Elements
* Household sampling
* Behavioral variation
* Weather noise (optional extension)

# Monte Carlo KPIs
For each run:
* Aggregate feeder kW series
* Peak MW & timing
* Hourly overload events

Across runs:
* Median KPIs
* Load-curve mean & variability band

🔧 ## Development Notes
* Designed for deployment via rsconnect-python to shinyapps.io
* Backend functions in src/ support use in notebooks & batch simulation
* No external proprietary datasets

📄 ## License

MIT License 

👤 ## Author

Tony Peluso, PhD
Energy Modelling & Grid Analytics — Montreal, QC
📧 tonympeluso@gmail.com

🔗 GitHub: https://github.com/TonyMPeluso

🔗 LinkedIn: https://www.linkedin.com/in/tony-peluso-phd
