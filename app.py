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


def export_to_pdf(bm_equity, asset_returns, kpi_rows, start_date, end_date, ytd_value=None):
    """
    Costruisce un report in una singola pagina PDF, formato A4 orizzontale,
    con i box KPI in alto e l'equity curve sotto. Uso matplotlib (già una
    dipendenza dell'app) invece di kaleido, così non serve installare
    pacchetti extra su Streamlit Cloud.

    Margini pensati per la stampa: non incollati al bordo pagina (rischio
    di taglio in stampa / etichette asse Y tagliate) ma comunque contenuti.
    Definiti a mano via GridSpec invece che con bbox_inches='tight' (che a
    volte aggiunge padding imprevedibile).

    ytd_value: rendimento YTD del benchmark (float, es. 0.0315), usato per
    evidenziare il tratto "da inizio anno" direttamente sulla curva.
    """
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch

    A4_LANDSCAPE = (11.69, 8.27)  # pollici
    fig = plt.figure(figsize=A4_LANDSCAPE)
    fig.patch.set_facecolor("white")

    # left=0.075 (non 0.035 come nella prima versione): con un margine
    # troppo stretto le etichette dell'asse Y ("Rendimento cumulato %")
    # finivano tagliate fuori dall'area stampabile.
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[0.55, 0.85, 4.6], hspace=0.35,
        left=0.075, right=0.97, top=0.93, bottom=0.13,
    )

    # ── Header ──
    fig.text(0.075, 0.965, "Benchmark Finanziario Personalizzato",
              fontsize=17, fontweight="bold", color="#0f172a")
    fig.text(0.075, 0.945,
              f"Periodo: {start_date} → {end_date}   ·   "
              f"Generato il {TODAY.strftime('%d/%m/%Y')}   ·   "
              f"Fonte: Yahoo Finance (EUR)",
              fontsize=8.5, color="#64748b")
    fig.add_artist(plt.Line2D([0.075, 0.97], [0.935, 0.935],
                              transform=fig.transFigure,
                              color="#e2e8f0", linewidth=1))

    # ── Riga KPI: card con bordo arrotondato, come sul cruscotto web ──
    ax_kpi = fig.add_subplot(gs[1])
    ax_kpi.axis("off")
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    n = len(kpi_rows)
    gap = 0.015
    card_w = (1 - gap * (n - 1)) / n
    for i, (label, val_str, is_positive) in enumerate(kpi_rows):
        x = i * (card_w + gap)
        color = "#111827"
        if is_positive is True:
            color = "#059669"
        elif is_positive is False:
            color = "#dc2626"
        ax_kpi.add_patch(FancyBboxPatch(
            (x, 0.05), card_w, 0.9,
            boxstyle="round,pad=0.006,rounding_size=0.03",
            linewidth=0.9, edgecolor="#e2e8f0", facecolor="#f8fafc",
            transform=ax_kpi.transAxes,
        ))
        cx = x + card_w / 2
        ax_kpi.text(cx, 0.66, label, ha="center", va="center",
                    fontsize=9.5, color="#64748b", transform=ax_kpi.transAxes)
        ax_kpi.text(cx, 0.30, val_str, ha="center", va="center",
                    fontsize=18, fontweight="bold", color=color,
                    transform=ax_kpi.transAxes)

    # ── Equity curve ──
    ax_chart = fig.add_subplot(gs[2])
    # Palette con un colore distinto per ciascun asset, per avere una
    # legenda completa e leggibile (prima erano tutti in grigio uniforme).
    palette = ["#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4",
              "#84cc16", "#ec4899", "#6366f1", "#14b8a6", "#f97316"]
    legend_handles = []
    for i, ticker in enumerate(asset_returns.columns):
        eq = (1 + asset_returns[ticker]).cumprod()
        color = palette[i % len(palette)]
        line, = ax_chart.plot(eq.index, (eq - 1) * 100, linewidth=1.1,
                              alpha=0.85, color=color,
                              label=LABELS.get(ticker, ticker))
        legend_handles.append(line)

    bm_perf = (bm_equity - 1) * 100
    bm_line, = ax_chart.plot(bm_equity.index, bm_perf, linewidth=2.6,
                             color=BM_COLOR, label="BENCHMARK", zorder=5)
    legend_handles.append(bm_line)
    ax_chart.axhline(0, linestyle="--", linewidth=0.8, color="#cbd5e1", zorder=1)

    # Evidenzia il tratto "da inizio anno" direttamente sulla linea del
    # benchmark, con un alone più spesso, due marker (inizio/fine YTD) e
    # un'etichetta con il valore — invece di una linea orizzontale generica
    # sganciata dalla curva.
    ytd_mask = bm_equity.index.year == TODAY.year
    if ytd_mask.any() and ytd_value is not None and pd.notna(ytd_value):
        seg_idx = bm_equity.index[ytd_mask]
        seg = bm_perf.loc[seg_idx]
        ax_chart.plot(seg.index, seg.values, linewidth=4.5, color=BM_COLOR,
                      alpha=0.18, zorder=4, solid_capstyle="round")
        ax_chart.scatter([seg.index[0], seg.index[-1]],
                         [seg.iloc[0], seg.iloc[-1]],
                         color=BM_COLOR, s=28, zorder=6,
                         edgecolor="white", linewidth=1.2)
        ax_chart.annotate(
            f"YTD {ytd_value:+.2%}",
            xy=(seg.index[-1], seg.iloc[-1]),
            xytext=(12, 14), textcoords="offset points",
            fontsize=9.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=BM_COLOR, edgecolor="none"),
            arrowprops=dict(arrowstyle="-", color=BM_COLOR, lw=1.2),
        )

    ax_chart.set_ylabel("Rendimento cumulato (%)", fontsize=9.5, color="#334155")
    ax_chart.tick_params(labelsize=8, colors="#475569")
    ax_chart.grid(alpha=0.2, linewidth=0.6)
    for spine in ["top", "right"]:
        ax_chart.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_chart.spines[spine].set_color("#cbd5e1")

    # Legenda completa con tutti gli indici + benchmark, sotto il grafico
    # (con 8+ asset non ci sta leggibile dentro l'area del grafico).
    fig.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.53, 0.015), ncol=min(len(legend_handles), 5),
              fontsize=8, frameon=False, columnspacing=1.4, handlelength=1.6)

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)
    return buf

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
            min_value=0.0,
            max_value=100.0,
            value=round(default_w * 100, 1),
            step=0.5,
            format="%.1f%%",
            key=f"w_{ticker}",
        )
 
    total_w = sum(weights_input.values())
 
    # FIX: tolleranza adeguata allo step 0.1%
    if abs(total_w - 100) > 0.05:
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

# FIX: senza session_state, cliccare QUALSIASI altro bottone della pagina
# (es. "Scarica report Excel") provoca un rerun in cui `run` torna False,
# facendo scattare st.stop() e resettando l'intera pagina alla schermata
# iniziale. Con session_state l'esito di "Esegui analisi" resta valido
# anche nei rerun successivi innescati da altri widget/bottoni.
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if run:
    st.session_state.analysis_done = True

if not st.session_state.analysis_done:
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
        # FIX: (1+returns).cumprod() è già una curva equity ancorata a t0.
        # Ridividere per eq.iloc[0] (come prima) cancellava il rendimento
        # del primo giorno e rendeva il grafico incoerente con i box KPI.
        eq = (1 + asset_returns[ticker]).cumprod()
        fig.add_trace(go.Scatter(
            x=eq.index, y=(eq - 1) * 100,
            name=LABELS.get(ticker, ticker),
            mode="lines", line=dict(width=1), opacity=0.4,
        ))
    fig.add_trace(go.Scatter(
        x=bm_equity.index, y=(bm_equity - 1) * 100,
        name="BENCHMARK", mode="lines",
        line=dict(width=3, color=BM_COLOR),
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#9ca3af")

    # Linea di riferimento YTD: il box "Rend. YTD" è calcolato sempre da
    # inizio anno solare, indipendentemente dal periodo selezionato in
    # sidebar. Quando il periodo scelto non è "YTD" (es. default 1Y), il
    # valore finale del grafico e il box YTD NON devono coincidere: sono
    # due metriche diverse per costruzione. Questa linea rende visibile
    # dove si colloca il rendimento YTD rispetto al periodo mostrato.
    ytd_val = metrics["Rendimento YTD"]
    if pd.notna(ytd_val):
        fig.add_hline(
            y=ytd_val * 100, line_width=1.5, line_dash="dot",
            line_color="#e3a008",
            annotation_text=f"YTD: {ytd_val:.2%}",
            annotation_position="bottom right",
            annotation_font_color="#e3a008",
        )

    fig.update_layout(
        title="Equity Curve — Benchmark vs Asset",
        yaxis_title="Rendimento cumulato (%)",
        template=TEMPLATE, height=480, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    if period_choice != "YTD":
        st.caption(
            "ℹ️ Il grafico mostra il rendimento del periodo selezionato "
            f"(**{period_choice}**), mentre il box 'Rend. YTD' mostra sempre "
            "il rendimento da inizio anno solare — per questo i due valori "
            "normalmente non coincidono (linea gialla tratteggiata = livello YTD)."
        )

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
        bm_perf = (bm_equity - 1) * 100  # FIX: nessuna doppia normalizzazione
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

# FIX: niente più "bottone dentro un bottone". Prima il download_button
# veniva creato solo DOPO il click sul primo bottone, e quel click
# provocava comunque un rerun in cui - senza session_state - la pagina
# si resettava (vedi fix più sopra). Ora i download_button sono generati
# direttamente: al click parte subito il download, senza doppio click.
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    excel_buffer = export_to_excel(
        prices, bm_returns, bm_equity,
        asset_metrics, contributions, weights,
    )
    st.download_button(
        label="📥 Scarica report Excel",
        data=excel_buffer,
        file_name=f"benchmark_{start_date}_{end_date}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with col_exp2:
    pdf_kpi_rows = [
        ("Rend. cumulato", fmt_pct(metrics["Rendimento cumulato"]),
         metrics["Rendimento cumulato"] > 0),
        ("Rend. YTD", fmt_pct(metrics["Rendimento YTD"]),
         metrics["Rendimento YTD"] > 0 if pd.notna(metrics["Rendimento YTD"]) else None),
        ("Volatilità ann.", fmt_pct(metrics["Volatilità (ann.)"]), None),
        ("Sharpe Ratio", fmt_num(metrics["Sharpe Ratio"]),
         metrics["Sharpe Ratio"] > 0 if pd.notna(metrics["Sharpe Ratio"]) else None),
        ("Max Drawdown", fmt_pct(metrics["Max Drawdown"]),
         metrics["Max Drawdown"] > 0 if pd.notna(metrics["Max Drawdown"]) else None),
    ]
    pdf_buffer = export_to_pdf(
        bm_equity, asset_returns, pdf_kpi_rows, start_date, end_date,
        ytd_value=metrics["Rendimento YTD"],
    )
    st.download_button(
        label="🖨️ Scarica report PDF (A4 orizzontale)",
        data=pdf_buffer,
        file_name=f"benchmark_{start_date}_{end_date}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
