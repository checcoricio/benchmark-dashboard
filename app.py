# =============================================================================
# app.py — Streamlit Dashboard Benchmark Finanziario
# Avvio: streamlit run app.py
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime
import io

from benchmark import (
    DEFAULT_PORTFOLIO, LABELS, ASSET_CLASSES, PERIODS, TODAY,
    validate_weights, download_prices, compute_benchmark,
    compute_metrics, compute_asset_contribution,
    compute_class_performance, export_to_excel,
)

# ─────────────────────────────────────────────────────────────────────────────
# SETUP PAGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Benchmark Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #64748b; margin-bottom: 4px; }
    .metric-value { font-size: 24px; font-weight: 600; color: #0f172a; }
    .metric-pos   { color: #059669; }
    .metric-neg   { color: #dc2626; }
    section[data-testid="stSidebar"] { background: #f1f5f9; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONTROLLI
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configurazione")

    # ── Periodo ──
    st.subheader("📅 Periodo")
    period_choice = st.radio(
        "Seleziona periodo",
        options=["YTD", "1M", "3M", "6M", "1Y", "3Y", "5Y", "Custom"],
        index=4,
        horizontal=False,
    )

    if period_choice == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data inizio", value=date(2026, 1, 1),
                                       max_value=TODAY)
        with col2:
            end_date = st.date_input("Data fine", value=TODAY,
                                     max_value=TODAY)
    else:
        start_date = PERIODS[period_choice]
        end_date   = TODAY

    st.caption(f"Dal {start_date} al {end_date}")

    st.divider()

    # ── Pesi ──
    st.subheader("⚖️ Pesi portafoglio")
    st.caption("Modifica i pesi — devono sommare a 100%")

    weights_input = {}
    for ticker, default_w in DEFAULT_PORTFOLIO.items():
        label = LABELS.get(ticker, ticker)
        weights_input[ticker] = st.slider(
            label,
            min_value=0,5,
            max_value=100,
            value=int(default_w * 100),
            step=1,
            format="%d%%",
            key=f"w_{ticker}",
        )

    total_w = sum(weights_input.values())

    # FIX: confronto corretto con 100 (i valori slider sono interi 0-100)
    if abs(total_w - 100) > 0.1:
        st.error(f"⚠️ Somma pesi: {total_w}% — deve essere 100%")
        weights_ok = False
    else:
        st.success(f"✅ Somma pesi: {total_w}%")
        weights_ok = True

    st.divider()

    # ── Risk free rate ──
    st.subheader("📐 Risk-free rate")
    rf_rate = st.slider("Tasso annuo (%)", 0.0, 10.0, 3.5, 0.1) / 100

    st.divider()

    run = st.button("▶  Esegui analisi", type="primary",
                    disabled=not weights_ok, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

st.title("📊 Benchmark Finanziario Personalizzato")
st.caption(f"Aggiornato al {TODAY.strftime('%d/%m/%Y')} · Dati: Yahoo Finance (Adj. Close, EUR)")

if not run:
    st.info("👈 Configura i parametri nella sidebar e premi **Esegui analisi**")
    st.stop()

# ── Download dati ──
with st.spinner("⬇️ Download prezzi in corso..."):
    try:
        prices = download_prices(
            list(weights_input.keys()),
            start_date,
            end_date,
        )
    except Exception as e:
        st.error(f"Errore download dati: {e}")
        st.stop()

# Normalizza pesi sugli asset effettivamente scaricati
# FIX: dividi per 100 per convertire da interi (es. 37) a frazioni (es. 0.37)
w_available = {k: v / 100 for k, v in weights_input.items() if k in prices.columns}
w_total = sum(w_available.values())
weights = {k: v / w_total for k, v in w_available.items()}

# ── Calcoli ──
bm_returns, bm_equity, asset_returns = compute_benchmark(prices, weights)
metrics       = compute_metrics(bm_returns, rf_rate)
asset_metrics = {t: compute_metrics(asset_returns[t], rf_rate) for t in asset_returns.columns}
contributions = compute_asset_contribution(asset_returns, weights)
class_equity  = compute_class_performance(asset_returns, weights)


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("📈 Metriche Benchmark")
c1, c2, c3, c4, c5 = st.columns(5)

def fmt_pct(v): return f"{v:.2%}" if pd.notna(v) else "N/A"
def fmt_num(v): return f"{v:.2f}" if pd.notna(v) else "N/A"
def color(v):   return "metric-pos" if v > 0 else "metric-neg"

kpis = [
    ("Rend. cumulato",  metrics["Rendimento cumulato"],  fmt_pct, True),
    ("Rend. YTD",       metrics["Rendimento YTD"],       fmt_pct, True),
    ("Volatilità ann.", metrics["Volatilità (ann.)"],    fmt_pct, False),
    ("Sharpe Ratio",    metrics["Sharpe Ratio"],         fmt_num, True),
    ("Max Drawdown",    metrics["Max Drawdown"],         fmt_pct, True),
]

for col, (label, val, formatter, use_color) in zip([c1,c2,c3,c4,c5], kpis):
    with col:
        css_color = color(val) if use_color and pd.notna(val) else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {css_color}">{formatter(val)}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# GRAFICI
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Equity Curve",
    "📉 Drawdown",
    "🌊 Rolling Volatility",
    "🔥 Correlazioni",
    "🥧 Composizione",
    "📋 KPI per Asset",
])

TEMPLATE = "plotly_white"
BM_COLOR = "#1a56db"

# ── Tab 1: Equity Curve ──
with tab1:
    fig = go.Figure()
    for ticker in asset_returns.columns:
        eq = (1 + asset_returns[ticker]).cumprod()
        eq = eq / eq.iloc[0]
        fig.add_trace(go.Scatter(
            x=eq.index, y=(eq - 1) * 100,
            name=LABELS.get(ticker, ticker),
            mode="lines", line=dict(width=1), opacity=0.4,
        ))
    bm_equity_norm = bm_equity / bm_equity.iloc[0]
    fig.add_trace(go.Scatter(
        x=bm_equity_norm.index, y=(bm_equity_norm - 1) * 100,
        name="BENCHMARK", mode="lines",
        line=dict(width=3, color=BM_COLOR),
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        title="Equity Curve — Benchmark vs Asset",
        yaxis_title="Rendimento cumulato (%)",
        template=TEMPLATE, height=480, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Asset class
    if not class_equity.empty:
        st.subheader("Performance per Asset Class")
        CLASS_COLORS = {"Equity": "#1a56db", "Fixed Income": "#0e9f6e",
                        "Mixed Allocation": "#e3a008", "Cash": "#6b7280"}
        fig2 = go.Figure()
        for col in class_equity.columns:
            perf = (class_equity[col] - 1) * 100
            fig2.add_trace(go.Scatter(
                x=perf.index, y=perf, name=col, mode="lines",
                line=dict(width=2.5, color=CLASS_COLORS.get(col)),
            ))
        bm_perf = (bm_equity / bm_equity.iloc[0] - 1) * 100
        fig2.add_trace(go.Scatter(
            x=bm_perf.index, y=bm_perf, name="BENCHMARK TOTALE",
            mode="lines", line=dict(width=3, color="#111827", dash="dot"),
        ))
        fig2.add_hline(y=0, line_width=1, line_dash="dash", line_color="#9ca3af")
        fig2.update_layout(
            yaxis_title="Rendimento cumulato (%)",
            template=TEMPLATE, height=380, hovermode="x unified",
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Drawdown ──
with tab2:
    equity     = (1 + bm_returns).cumprod()
    rolling_mx = equity.cummax()
    drawdown   = ((equity - rolling_mx) / rolling_mx) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        fill="tozeroy", fillcolor="rgba(220, 38, 38, 0.15)",
        line=dict(color="#dc2626", width=1.5), name="Drawdown",
    ))
    fig.update_layout(
        title="Drawdown — Benchmark",
        yaxis_title="Drawdown (%)", template=TEMPLATE,
        height=380, hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Max Drawdown: **{metrics['Max Drawdown']:.2%}** in data {metrics['Data Max DD']}")

# ── Tab 3: Rolling Volatility ──
with tab3:
    rv21 = bm_returns.rolling(21).std() * np.sqrt(252) * 100
    rv63 = bm_returns.rolling(63).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rv21.index, y=rv21, name="Vol. 21d (1M)",
                             line=dict(width=1.5, color="#f05252")))
    fig.add_trace(go.Scatter(x=rv63.index, y=rv63, name="Vol. 63d (3M)",
                             line=dict(width=2, color=BM_COLOR)))
    fig.update_layout(
        title="Rolling Volatility Annualizzata",
        yaxis_title="Volatilità (%)", template=TEMPLATE,
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Correlazioni ──
with tab4:
    corr = asset_returns.rename(columns=LABELS).corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=-1, vmax=1,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice di Correlazione", fontsize=13, pad=10)
    plt.tight_layout()
    st.pyplot(fig)

# ── Tab 5: Composizione ──
with tab5:
    col_pie, col_bar = st.columns(2)
    tickers_list = list(weights.keys())
    pesi_list    = [weights[t] * 100 for t in tickers_list]
    nomi_list    = [LABELS.get(t, t) for t in tickers_list]

    with col_pie:
        fig = go.Figure(go.Pie(
            labels=nomi_list, values=pesi_list,
            hole=0.38, textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig.update_layout(title="Allocazione %", height=400,
                          showlegend=False, template=TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with col_bar:
        idx = np.argsort(pesi_list)[::-1]
        fig = go.Figure(go.Bar(
            x=[nomi_list[i] for i in idx],
            y=[pesi_list[i] for i in idx],
            marker_color=BM_COLOR,
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(title="Pesi ordinati", height=400,
                          yaxis_title="%", template=TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 6: KPI per Asset ──
with tab6:
    rows = []
    for ticker in asset_returns.columns:
        m = asset_metrics[ticker]
        rows.append({
            "Asset":      LABELS.get(ticker, ticker),
            "Peso %":     f"{weights.get(ticker, 0):.2%}",
            "Rend. cum.": f"{m['Rendimento cumulato']:.2%}",
            "Vol. ann.":  f"{m['Volatilità (ann.)']:.2%}",
            "Sharpe":     f"{m['Sharpe Ratio']:.2f}",
            "Max DD":     f"{m['Max Drawdown']:.2%}",
            "Calmar":     f"{m['Calmar Ratio']:.2f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Contributo alla performance")
    st.dataframe(contributions, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT EXCEL
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("⬇️ Export")

if st.button("📥 Scarica report Excel", use_container_width=False):
    filename = export_to_excel(
        prices, bm_returns, bm_equity,
        asset_metrics, contributions, weights,
    )
    with open(filename, "rb") as f:
        st.download_button(
            label="💾 Clicca qui per scaricare",
            data=f,
            file_name=f"benchmark_{start_date}_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
