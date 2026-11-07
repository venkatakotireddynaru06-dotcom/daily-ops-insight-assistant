"""Pilot evaluation: writes docs/evaluation_results.md with measured numbers."""
from __future__ import annotations
import time
from .analysis import OpsAnalyzer, load_data
from .config import METRICS, ROOT
from .report import build_weekly_report, save_report

MANUAL_MIN_PER_WEEK = 45  # ASSUMPTION - replace with the figure from a real frontline interview


def run() -> str:
    df = load_data()
    an = OpsAnalyzer(df)
    rows = []
    for t, m in METRICS.items():
        b = an.backtest(t)
        gain = (1 - b["mape_model"] / b["mape_naive"]) * 100
        rows.append(f"| {m['label_en']} | {b['mape_naive']}% | {b['mape_model']}% | {gain:.0f}% |")

    flagged = set(an.anomalies("park_visitors").date)
    events = set(df[df.special_event == 1].date)
    tp = len(flagged & events)
    recall, precision = tp / len(events), (tp / len(flagged) if flagged else 0)

    t0 = time.perf_counter()
    res = build_weekly_report(an)
    save_report(res)
    secs = time.perf_counter() - t0
    saved = MANUAL_MIN_PER_WEEK - secs / 60

    out = f"""# Pilot Evaluation Results (auto-generated)

Data: synthetic, {len(df)} days. Forecast test window: last 90 days (never seen in training).

## 1. Forecast accuracy (MAPE, lower is better)
| Metric | Baseline: same weekday last week | Model | Improvement |
|---|---|---|---|
""" + "\n".join(rows) + f"""

## 2. Anomaly detection (park visitors, validated against hidden event days)
- Hidden event days: {len(events)} | flagged days: {len(flagged)}
- Recall: {recall:.0%} | Precision: {precision:.0%}

## 3. Report automation
- Automated weekly report generation (pipeline + analysis + Excel/Markdown): {secs:.1f} s
- Assumed manual effort: {MANUAL_MIN_PER_WEEK} min/week (**assumption: replace with real interview data**)
- Estimated time saved: ~{saved:.0f} min/week per report owner (~{saved * 52 / 60:.0f} h/year)

## 4. Caveats
- Synthetic data: accuracy numbers show the method works, not what real operations would achieve.
- Forecast uses seasonal-average weather; real use would plug in a weather forecast feed.
"""
    (ROOT / "docs" / "evaluation_results.md").write_text(out, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(run())
