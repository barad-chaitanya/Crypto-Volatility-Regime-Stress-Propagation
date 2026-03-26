Project Overview
This project is an advanced quant research dashboard for analyzing crypto market volatility regimes and visualizing stress propagation across assets during crash events. It is designed for institutional-grade analytics and interactive visualization.

Key Features
Multi-asset data ingestion: Upload CSV or fetch live prices via Yahoo Finance or CCXT.
Volatility regime detection: Clustering and threshold-based analysis.
Stress propagation metrics: Network graph of asset correlations during stress.
Systemic risk index: Composite quant metric for market risk.
Cinematic dashboard: Plotly-powered, dark crisis-themed visuals, interactive cards, and animated charts.
File Structure
app.py : Main Streamlit application (single-file, modular functions)
.venv/ : Python virtual environment (auto-created)
README.md : Project documentation (this file)
How to Run / Fork
Fork or clone this repo to your local machine.
Install Python 3.8+ (recommended: 3.10+).
Create a virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies:
pip install streamlit pandas numpy plotly scikit-learn scipy yfinance ccxt networkx
Run the app:
streamlit run app.py
Open the dashboard in your browser (URL shown in terminal).
How to Use
Use the sidebar to select data source, set analysis parameters, and upload or fetch data.
Explore volatility surfaces, animated heatmaps, stress propagation networks, and systemic risk metrics.
All code is modular and can be extended for further quant research.
