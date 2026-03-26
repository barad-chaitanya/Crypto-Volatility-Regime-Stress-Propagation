import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.sparse.csgraph import shortest_path
from scipy.stats import zscore
import yfinance as yf
import ccxt
import datetime

# =============================
# CONFIG & STYLES
# =============================
st.set_page_config(
    page_title="Crypto Volatility Regime & Stress Propagation Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

DARK_BG = "#18181b"
STRESS_COLOR = "#ff0033"
VOL_COLOR = "#00e6e6"
RISK_COLOR = "#ffae00"

st.markdown(
    f"""
    <style>
    .reportview-container {{ background: {DARK_BG}; }}
    .sidebar .sidebar-content {{ background: #111112; }}
    .block-container {{ padding-top: 2rem; }}
    .metric-card {{ background: #222226; color: #fff; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 0 24px #000a; }}
    .stButton>button {{ background: {STRESS_COLOR}; color: #fff; border-radius: 8px; }}
    .stSlider {{ color: {VOL_COLOR}; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# SIDEBAR CONTROLS
# =============================
st.sidebar.title("Lab Controls")
vol_window = st.sidebar.slider("Volatility Window (days)", 5, 60, 21, 1)
regime_sensitivity = st.sidebar.slider("Regime Sensitivity", 1, 10, 4, 1)
stress_threshold = st.sidebar.slider("Stress Threshold (z-score)", 1.0, 4.0, 2.5, 0.1)

# =============================
# DATA PIPELINE
# =============================
def load_csv():
    uploaded = st.sidebar.file_uploader("Upload CSV (multi-asset prices)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded, index_col=0, parse_dates=True)
        return df
    return None

def load_yfinance(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    # Handle both single and multi-ticker cases
    if "Adj Close" in data.columns:
        # Single ticker: columns are ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df = data[["Adj Close"]].rename(columns={"Adj Close": tickers if isinstance(tickers, str) else tickers[0]})
    elif isinstance(data.columns, pd.MultiIndex):
        # Multi-ticker: columns are MultiIndex (field, ticker)
        df = data["Adj Close"]
    else:
        st.error("Could not find 'Adj Close' in downloaded data. Check tickers and date range.")
        st.stop()
    return df

def load_ccxt(symbols, exchange_name="binance", since_days=180):
    exchange = getattr(ccxt, exchange_name)()
    dfs = []
    now = int(datetime.datetime.now().timestamp() * 1000)
    since = now - since_days * 24 * 60 * 60 * 1000
    for symbol in symbols:
        ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("date", inplace=True)
        dfs.append(df[["close"]].rename(columns={"close": symbol}))
    return pd.concat(dfs, axis=1)

# =============================
# DATA INGESTION
# =============================
st.sidebar.subheader("Data Source")
data_source = st.sidebar.selectbox("Select Data Source", ["CSV Upload", "Yahoo Finance", "CCXT Live"])

if data_source == "CSV Upload":
    price_df = load_csv()
    asset_list = list(price_df.columns) if price_df is not None else []
elif data_source == "Yahoo Finance":
    tickers = st.sidebar.text_input("Tickers (comma separated)", "BTC-USD,ETH-USD,SOL-USD")
    start = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=180))
    end = st.sidebar.date_input("End Date", datetime.date.today())
    asset_list = [t.strip() for t in tickers.split(",")]
    price_df = load_yfinance(asset_list, start, end)
elif data_source == "CCXT Live":
    symbols = st.sidebar.text_input("Symbols (comma separated)", "BTC/USDT,ETH/USDT,SOL/USDT")
    asset_list = [s.strip() for s in symbols.split(",")]
    price_df = load_ccxt(asset_list)
else:
    price_df = None
    asset_list = []

if price_df is None or len(asset_list) == 0:
    st.warning("Please upload or select valid data.")
    st.stop()

# =============================
# FEATURE ENGINEERING
# =============================
def compute_log_returns(df):
    return np.log(df / df.shift(1)).dropna()

def compute_rolling_vol(df, window):
    return df.rolling(window).std() * np.sqrt(252)

def compute_rolling_corr(df, window):
    return df.rolling(window).corr()

def compute_vol_clustering(vol_df):
    # Use autocorrelation of volatility as clustering metric
    return vol_df.apply(lambda x: x.autocorr(1))

returns_df = compute_log_returns(price_df)
vol_df = compute_rolling_vol(returns_df, vol_window)
corr_df = compute_rolling_corr(returns_df, vol_window)
vol_cluster_metric = compute_vol_clustering(vol_df)

# =============================
# VOLATILITY REGIME DETECTION
# =============================
def detect_vol_regimes(vol_df, sensitivity):
    scaler = StandardScaler()
    X = scaler.fit_transform(vol_df.fillna(0))
    n_clusters = min(max(2, sensitivity), 6)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

regime_labels = detect_vol_regimes(vol_df, regime_sensitivity)
vol_df["Regime"] = regime_labels

# =============================
# STRESS & PROPAGATION METRICS
# =============================
def detect_stress_periods(vol_df, threshold):
    z = zscore(vol_df.fillna(0))
    stress_mask = (np.abs(z) > threshold)
    return stress_mask

stress_mask = detect_stress_periods(vol_df[asset_list], stress_threshold)

# Correlation spikes
corr_spikes = corr_df.groupby(level=0).apply(lambda x: (x > 0.7).sum().sum())

# Stress propagation: shortest path in correlation graph
corr_matrix = corr_df.iloc[-1].fillna(0)
if corr_matrix.values.ndim == 2:
    try:
        np.fill_diagonal(corr_matrix.values, 0)
        propagation_graph = np.abs(corr_matrix.values)
        propagation_dist = shortest_path(propagation_graph, directed=False)
    except Exception as e:
        st.warning(f"Could not compute stress propagation graph: {e}")
        propagation_graph = None
        propagation_dist = None
else:
    propagation_graph = None
    propagation_dist = None

# Systemic risk index: weighted sum of vol, corr spikes, clustering
risk_index = (
    vol_df[asset_list].mean(axis=1).fillna(0) * 0.5 +
    corr_spikes.fillna(0) * 0.3 +
    vol_cluster_metric.mean() * 0.2
)

# =============================
# METRIC CARDS
# =============================
st.markdown("<h1 style='color:#fff;font-size:2.8rem;font-weight:700;'>Crypto Volatility Regime & Stress Propagation Lab</h1>", unsafe_allow_html=True)
st.markdown("<div style='height:18px'></div>")
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<div class='metric-card'><h2>Avg Volatility</h2><span style='font-size:2.2rem;color:{VOL_COLOR};'>{vol_df[asset_list].mean().mean():.3f}</span></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><h2>Max Stress</h2><span style='font-size:2.2rem;color:{STRESS_COLOR};'>{stress_mask.sum().sum()}</span></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><h2>Corr Spikes</h2><span style='font-size:2.2rem;color:{RISK_COLOR};'>{corr_spikes.max():.0f}</span></div>", unsafe_allow_html=True)
col4.markdown(f"<div class='metric-card'><h2>Systemic Risk</h2><span style='font-size:2.2rem;color:#fff;background:{RISK_COLOR};border-radius:8px;padding:0.2em 0.6em;'>{risk_index.max():.2f}</span></div>", unsafe_allow_html=True)

# =============================
# 3D VOLATILITY SURFACE
# =============================
vol_surface = go.Surface(
    z=vol_df[asset_list].values.T,
    x=vol_df.index,
    y=asset_list,
    colorscale=[[0, VOL_COLOR], [1, STRESS_COLOR]],
    showscale=True,
    hovertemplate="<b>%{y}</b><br>Time: %{x}<br>Volatility: %{z:.3f}<extra></extra>"
)
fig_vol = go.Figure(data=[vol_surface])
fig_vol.update_layout(
    title="3D Volatility Surface",
    autosize=True,
    template="plotly_dark",
    margin=dict(l=0, r=0, b=0, t=40),
    scene=dict(
        xaxis_title="Time",
        yaxis_title="Asset",
        zaxis_title="Volatility",
        bgcolor=DARK_BG
    ),
    height=600
)
st.plotly_chart(fig_vol, use_container_width=True)

# =============================
# ANIMATED CORRELATION HEATMAP
# =============================
frames = []
for i, idx in enumerate(corr_df.index.levels[0]):
    mat = corr_df.loc[idx].fillna(0)
    frame = go.Heatmap(
        z=mat.values,
        x=mat.columns,
        y=mat.index,
        colorscale=[[0, DARK_BG], [0.5, VOL_COLOR], [1, STRESS_COLOR]],
        zmin=-1, zmax=1,
        hovertemplate="<b>%{y}-%{x}</b><br>Corr: %{z:.2f}<extra></extra>"
    )
    frames.append(frame)
fig_corr = go.Figure(frames)
fig_corr.update_layout(
    title="Animated Correlation Heatmap",
    template="plotly_dark",
    margin=dict(l=0, r=0, b=0, t=40),
    height=500,
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)
st.plotly_chart(fig_corr, use_container_width=True)

# =============================
# STRESS PROPAGATION NETWORK
# =============================
import networkx as nx
if propagation_graph is not None:
    G = nx.Graph()
    for i, asset in enumerate(asset_list):
        G.add_node(asset)
    for i in range(len(asset_list)):
        for j in range(i+1, len(asset_list)):
            weight = corr_matrix.iloc[i, j]
            if abs(weight) > 0.5:
                G.add_edge(asset_list[i], asset_list[j], weight=weight)
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    edge_colors = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_colors.append(STRESS_COLOR if edge[2]['weight'] > 0.7 else VOL_COLOR)
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color=STRESS_COLOR),
        hoverinfo='none', mode='lines'))
    fig_net.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=28, color=VOL_COLOR),
        text=list(G.nodes()),
        textposition="bottom center",
        hoverinfo='text',
        hovertext=[f"{n}" for n in G.nodes()]
    ))
    fig_net.update_layout(
        title="Stress Propagation Network",
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40),
        height=500,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    st.plotly_chart(fig_net, use_container_width=True)
else:
    st.warning("Stress propagation network graph could not be loaded due to insufficient correlation data.")

# =============================
# SYSTEMIC RISK INDEX TIMESERIES
# =============================
fig_risk = go.Figure()
fig_risk.add_trace(go.Scatter(
    x=risk_index.index,
    y=risk_index.values,
    mode='lines',
    line=dict(color=RISK_COLOR, width=4),
    name='Systemic Risk Index',
    hovertemplate="Time: %{x}<br>Risk: %{y:.2f}<extra></extra>"
))
crisis_zones = risk_index > risk_index.mean() + risk_index.std()
fig_risk.add_trace(go.Scatter(
    x=risk_index.index[crisis_zones],
    y=risk_index.values[crisis_zones],
    mode='markers',
    marker=dict(size=12, color=STRESS_COLOR, symbol='star'),
    name='Crisis Zone',
    hovertemplate="<b>Crisis!</b><br>Time: %{x}<br>Risk: %{y:.2f}<extra></extra>"
))
fig_risk.update_layout(
    title="Systemic Risk Index (Crisis Highlighted)",
    template="plotly_dark",
    margin=dict(l=0, r=0, b=0, t=40),
    height=400,
    xaxis_title="Time",
    yaxis_title="Risk Index"
)
st.plotly_chart(fig_risk, use_container_width=True)

# =============================
# FOOTER
# =============================
st.markdown("<div style='height:32px'></div>")
st.markdown("<h3 style='color:#fff;text-align:center;'>Quant Research Software - Crypto Volatility Regime & Stress Propagation Lab</h3>", unsafe_allow_html=True)