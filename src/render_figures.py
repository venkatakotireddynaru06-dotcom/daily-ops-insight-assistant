"""Render presentation-quality figures to docs/images/ (used in README and slides).

    python -m src.render_figures
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .analysis import OpsAnalyzer, load_data
from .config import METRICS, ROOT

OUT = ROOT / "docs" / "images"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BLUE, ORANGE, GREY, RED = "#1F6FB2", "#E8833A", "#9AA5B1", "#D64545"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def surface(an, metric="park_visitors"):
    p = an.df.groupby(["dow", "month"])[metric].mean().unstack()
    X, Y = np.meshgrid(p.columns, p.index)
    fig = plt.figure(figsize=(9, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    s = ax.plot_surface(X, Y, p.values, cmap="YlGnBu", edgecolor="white", linewidth=0.3, antialiased=True)
    ax.set_xticks(range(1, 13)); ax.set_yticks(range(7)); ax.set_yticklabels(DOW)
    ax.set_xlabel("Month", labelpad=8); ax.set_zlabel("Avg visitors / day", labelpad=8)
    ax.view_init(28, -58)
    fig.colorbar(s, shrink=0.55, pad=0.08)
    ax.set_title("Demand landscape: month x weekday (park visitors)", fontsize=13, weight="bold")
    fig.savefig(OUT / "3d_demand_surface.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def scatter(an, metric="park_visitors"):
    d = an.df.sample(700, random_state=1)
    fig = plt.figure(figsize=(9, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(d.temp_c, d.rain_mm, d[metric], c=d[metric], cmap="YlGnBu", s=16, alpha=0.85, edgecolor="none")
    ax.set_xlabel("Temperature (C)", labelpad=8); ax.set_ylabel("Rain (mm)", labelpad=8); ax.set_zlabel("Visitors", labelpad=8)
    ax.view_init(22, -50)
    fig.colorbar(sc, shrink=0.55, pad=0.08)
    ax.set_title("Weather effect on park demand (700 sampled days)", fontsize=13, weight="bold")
    fig.savefig(OUT / "3d_weather_scatter.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def forecast(an, metric="park_visitors"):
    d = an.df.tail(90); f = an.forecast(metric, 7); a = an.anomalies(metric, start=d.date.min())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d.date, d[metric], color=BLUE, lw=1.6, label="Actual")
    ax.plot(d.date, an.oof(metric).loc[d.index], color=GREY, ls=":", lw=1.4, label="Expected (model)")
    ax.plot(f.date, f.forecast, color=ORANGE, lw=2.8, label="7-day forecast")
    ax.fill_between(f.date, f.low, f.high, color=ORANGE, alpha=0.2, label="80% range")
    ax.scatter(a.date, a.actual, marker="D", s=55, color=RED, edgecolor="white", zorder=5, label="Unusual day")
    ax.set_title("Park visitors: actual, expected, forecast and unusual days", weight="bold", loc="left")
    ax.grid(alpha=0.25); ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.savefig(OUT / "forecast.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def backtest(an):
    bt = [an.backtest(t) for t in METRICS]
    names = [METRICS[b["metric"]]["label_en"] for b in bt]; x = np.arange(len(bt))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, [b["mape_naive"] for b in bt], 0.4, color=GREY, label="Baseline (same weekday last week)")
    ax.bar(x + 0.2, [b["mape_model"] for b in bt], 0.4, color=BLUE, label="Model")
    for i, b in enumerate(bt):
        ax.text(i + 0.2, b["mape_model"] + 0.3, f"-{(1 - b['mape_model'] / b['mape_naive']) * 100:.0f}%", ha="center", color=BLUE, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("MAPE % (lower is better)")
    ax.set_title("Forecast error on held-out 90 days (synthetic data)", weight="bold", loc="left")
    ax.legend(frameon=False); ax.grid(axis="y", alpha=0.25)
    fig.savefig(OUT / "backtest.png", dpi=170, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    an = OpsAnalyzer(load_data())
    surface(an); scatter(an); forecast(an); backtest(an)
    print("Rendered:", *sorted(p.name for p in OUT.glob("*.png")))
