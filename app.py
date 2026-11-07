"""Daily Ops Insight Assistant - Streamlit dashboard.   Run: streamlit run app.py"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent import OpsAgent
from src.analysis import OpsAnalyzer, load_data
from src.config import METRICS
from src.report import build_weekly_report, to_excel_bytes

st.set_page_config(page_title="Daily Ops Insight Assistant", page_icon="🚃", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1250px}
.hero{background:linear-gradient(90deg,#12355B,#1F6FB2);color:#fff;padding:1.2rem 1.6rem;border-radius:12px;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.7rem;color:#fff;padding:0}.hero p{margin:.35rem 0 .6rem;opacity:.92}
.badge{display:inline-block;background:rgba(255,255,255,.18);padding:2px 10px;border-radius:999px;font-size:.75rem;margin-right:6px}
div[data-testid="stMetric"]{background:#F4F7FB;border:1px solid #E1E8F0;border-radius:10px;padding:.8rem 1rem}
</style>""", unsafe_allow_html=True)

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLORS = dict(actual="#1F6FB2", expected="#9AA5B1", forecast="#E8833A", flag="#D64545")


@st.cache_resource
def get_analyzer():
    return OpsAnalyzer(load_data())


an = get_analyzer()
last = an.df.date.max()
label = {k: f"{v['label_en']} / {v['label_ja']}" for k, v in METRICS.items()}

st.markdown("""<div class="hero"><h1>🚃 Daily Ops Insight Assistant</h1>
<p>From manual reporting to forecasts, automated reports and a data-grounded AI assistant, for railway, bus, park and hotel operations.</p>
<span class="badge">Python</span><span class="badge">scikit-learn</span><span class="badge">Claude tool use</span>
<span class="badge">Streamlit</span><span class="badge">Synthetic demo data</span></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Controls")
    metric = st.selectbox("Metric / 指標", list(METRICS), format_func=label.get)
    days = st.slider("History to show (days)", 30, 365, 120)
    st.divider()
    st.caption(f"Data through **{last.date()}**  \n{len(an.df):,} days · 4 business lines")
    st.caption("All data is synthetic. No personal data is used.")

t_dash, t_3d, t_fc, t_rep, t_ai, t_about = st.tabs(["📊 Dashboard", "🧊 3D explorer", "🎯 Forecast quality", "📄 Weekly report", "💬 AI assistant", "ℹ️ About"])

with t_dash:
    cols = st.columns(len(METRICS))
    for c, t in zip(cols, METRICS):
        k = an.kpis(t, last - pd.Timedelta(days=6), last)
        val = f"{k['value']:,.0f}" if METRICS[t]["agg"] == "sum" else f"{k['value']:.1f}%"
        c.metric(METRICS[t]["label_en"] + " · 7 days", val, None if k["change_pct"] is None else f"{k['change_pct']:+.1f}% vs prior week")
    hist = an.df[an.df.date > last - pd.Timedelta(days=days)]
    fc = an.forecast(metric, 7)
    pts = an.anomalies(metric, start=hist.date.min())
    fig = go.Figure()
    fig.add_scatter(x=hist.date, y=hist[metric], name="Actual", line=dict(width=1.8, color=COLORS["actual"]))
    fig.add_scatter(x=hist.date, y=an.oof(metric).loc[hist.index], name="Expected (model)", line=dict(dash="dot", color=COLORS["expected"]))
    fig.add_scatter(x=fc.date, y=fc.forecast, name="7-day forecast", line=dict(width=3, color=COLORS["forecast"]))
    fig.add_scatter(x=list(fc.date) + list(fc.date[::-1]), y=list(fc.high) + list(fc.low[::-1]), fill="toself",
                    fillcolor="rgba(232,131,58,.18)", line=dict(width=0), name="80% range", hoverinfo="skip")
    fig.add_scatter(x=pts.date, y=pts.actual, mode="markers", name="Unusual day",
                    marker=dict(size=11, symbol="diamond", color=COLORS["flag"], line=dict(width=1, color="white")))
    fig.update_layout(template="plotly_white", height=440, title=label[metric], margin=dict(l=0, r=0, t=50, b=0),
                      legend=dict(orientation="h", y=-0.12), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Forecast uses seasonal-average weather. Red diamonds: days far from what calendar and weather predict, worth asking frontline staff about.")
    if len(pts):
        with st.expander(f"Unusual days in view ({len(pts)})"):
            st.dataframe(pts.assign(date=pts.date.dt.date), hide_index=True, width="stretch")

with t_3d:
    st.caption("Drag to rotate · scroll to zoom · hover for values. Metric follows the sidebar selection.")
    p = an.df.groupby(["dow", "month"])[metric].mean().unstack()
    surf = go.Figure(go.Surface(z=p.values, x=list(p.columns), y=list(p.index), colorscale="YlGnBu", colorbar=dict(title="avg", len=0.6)))
    surf.update_layout(title="Demand landscape: month × weekday", height=540, margin=dict(l=0, r=0, t=50, b=0),
                       scene=dict(xaxis_title="Month", yaxis=dict(title="", tickvals=list(range(7)), ticktext=DOW), zaxis_title=METRICS[metric]["label_en"],
                                  camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))))
    st.plotly_chart(surf, width="stretch")
    st.markdown("**Reading it:** ridges show weekends and holidays; the swell shows summer seasonality. This is the pattern the forecast model learns.")
    smp = an.df.sample(600, random_state=1)
    sc = go.Figure(go.Scatter3d(x=smp.temp_c, y=smp.rain_mm, z=smp[metric], mode="markers",
                                marker=dict(size=3.5, color=smp[metric], colorscale="YlGnBu", opacity=0.85, colorbar=dict(title="value", len=0.6)),
                                text=smp.date.dt.strftime("%Y-%m-%d"), hovertemplate="%{text}<br>%{x:.0f}°C · %{y:.0f} mm<br>%{z:,.0f}<extra></extra>"))
    sc.update_layout(title="Weather effect: temperature × rain × demand (600 sampled days)", height=540, margin=dict(l=0, r=0, t=50, b=0),
                     scene=dict(xaxis_title="Temp (°C)", yaxis_title="Rain (mm)", zaxis_title=METRICS[metric]["label_en"]))
    st.plotly_chart(sc, width="stretch")

with t_fc:
    bt = pd.DataFrame([an.backtest(t) for t in METRICS])
    bt["Metric"] = bt.metric.map(lambda t: METRICS[t]["label_en"])
    bt["Improvement"] = ((1 - bt.mape_model / bt.mape_naive) * 100).round(0).astype(int).astype(str) + "%"
    st.markdown("**Backtest on the last 90 days (never seen during training).** Baseline = *same weekday last week*, the rule of thumb many teams use today.")
    f2 = go.Figure()
    f2.add_bar(x=bt.Metric, y=bt.mape_naive, name="Baseline", marker_color=COLORS["expected"])
    f2.add_bar(x=bt.Metric, y=bt.mape_model, name="Model", marker_color=COLORS["actual"])
    f2.update_layout(template="plotly_white", barmode="group", height=360, yaxis_title="MAPE % (lower is better)", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(f2, width="stretch")
    st.dataframe(bt[["Metric", "mape_naive", "mape_model", "Improvement", "mae_naive", "mae_model"]]
                 .rename(columns={"mape_naive": "Baseline MAPE %", "mape_model": "Model MAPE %", "mae_naive": "Baseline MAE", "mae_model": "Model MAE"}),
                 hide_index=True, width="stretch")
    st.info("Results are on synthetic data: they demonstrate the method, not real-world performance.")

with t_rep:
    c1, c2 = st.columns(2)
    end = c1.date_input("Week ending", value=last.date(), min_value=(an.df.date.min() + pd.Timedelta(days=13)).date(), max_value=last.date())
    lang = c2.radio("Language / 言語", ["en", "ja"], horizontal=True)
    res = build_weekly_report(an, end, lang)
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇ Markdown", res["markdown"], f"weekly_report_{end}.md", "text/markdown")
    d2.download_button("⬇ Excel", to_excel_bytes(res), f"weekly_report_{end}.xlsx")
    with st.container(border=True):
        st.markdown(res["markdown"])

with t_ai:
    agent = OpsAgent(an)
    st.caption(f"Mode: {agent.mode}. Every number comes from analysis tools, never from the model's memory.")
    st.markdown("Try: *Why was the park so busy recently?* · *2025-12-20 なぜ遊園地は少なかった?* · *hotel forecast* · *bus anomalies*")
    st.session_state.setdefault("chat", [])
    for role, text in st.session_state.chat:
        st.chat_message(role).write(text)
    if q := st.chat_input("Ask about operations / 質問してください"):
        st.session_state.chat.append(("user", q))
        st.chat_message("user").write(q)
        with st.spinner("Analyzing..."):
            a = agent.answer(q)
        st.session_state.chat.append(("assistant", a))
        st.chat_message("assistant").write(a)

with t_about:
    st.subheader("The problem")
    st.write("Operations staff spend hours every week copying numbers into spreadsheets, have no easy demand outlook, "
             "and cannot quickly ask *why* a day was unusual. *(Assumption modeled on typical railway, park and hotel reporting; to be validated in frontline interviews.)*")
    st.subheader("The approach: a small pilot, measured")
    st.graphviz_chart('''digraph{rankdir=LR;node[shape=box,style="rounded,filled",fillcolor="#EAF2FB",color="#1F6FB2",fontname="Helvetica"];
    D[label="Daily ops data\\n+ weather + holidays"];M[label="Forecast &\\nanomaly model"];R[label="Weekly report\\n(MD / Excel, EN/JA)"];
    T[label="Analysis tools"];A[label="AI assistant\\n(Claude tool use)"];U[label="Dashboard"];
    D->M->T;T->R;T->A;T->U;M->U}''')
    a1, a2, a3 = st.columns(3)
    a1.metric("Forecast error vs baseline", f"-{(1 - bt.mape_model.mean() / bt.mape_naive.mean()) * 100:.0f}%", "avg across 4 metrics")
    a2.metric("Weekly report", "one click", "was manual copy-paste")
    a3.metric("Agent hallucinated numbers", "0 by design", "tool-only answers")
    st.subheader("Design principles")
    st.markdown("- **Problem first, model second:** start from a reporting pain point.\n- **Honest baselines and caveats:** held-out test, synthetic data flagged everywhere.\n"
                "- **Grounded AI:** the LLM picks tools and phrases answers; it never computes or recalls numbers.\n"
                "- **Unexplained is allowed:** unexplained gaps are routed to frontline staff.\n- **Security:** aggregates only, no personal data, key via environment variable.")
    st.subheader("Next steps")
    st.markdown("1. Frontline interviews and one real anonymized export  2. Two-week parallel run with the manual report  "
                "3. Weather-forecast feed  4. Access control and approved LLM vendor  5. Rollout plan and training")
