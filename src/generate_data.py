"""Synthetic daily operations data for a tourism/transport group (clearly NOT real company data).

Patterns baked in: seasonality, weekend/holiday uplift, rain and cold penalties, and hidden
'special_event' days (used only to validate the anomaly detector, never as a model feature).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .calendar_jp import is_holiday
from .config import DATA_PATH


def generate(start="2023-01-01", end="2025-12-31", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end)
    n = len(dates)
    doy, dow = dates.dayofyear.values, dates.dayofweek.values
    hol = np.array([is_holiday(d) for d in dates], dtype=int)
    weekend = (dow >= 5).astype(int)
    off = np.maximum(hol, weekend)

    temp = 14 - 12 * np.cos(2 * np.pi * (doy - 20) / 365.25) + rng.normal(0, 3, n)
    summer = (np.sin(2 * np.pi * (doy - 100) / 365.25) + 1) / 2
    rainy = rng.random(n) < (0.18 + 0.15 * summer)
    rain = np.where(rainy, rng.gamma(2.0, 7.0, n), 0.0)
    event = np.zeros(n, dtype=int)
    event[rng.choice(n, size=int(n / 365 * 8), replace=False)] = 1

    park = 6500 * (1 + 0.30 * summer) * np.where(off == 1, 1.6, 1.0)
    park = park * (1 - np.clip(rain / 50, 0, 0.55)) * np.where(temp < 3, 0.8, 1.0)
    park = park * np.where(event == 1, 1.4, 1.0) * rng.lognormal(0, 0.07, n)

    occ = (60 + 8 * summer + np.where(dow == 5, 22, 0) + np.where(dow == 4, 8, 0)
           + np.where(hol == 1, 10, 0) - np.where(rain > 10, 3, 0)
           + np.where(event == 1, 6, 0) + rng.normal(0, 4, n))
    rail = 4800 + 0.55 * park + np.where(off == 0, 1600, 0) + rng.normal(0, 300, n)
    bus = 3200 + 0.22 * park + np.where(rain > 5, 350, 0) + rng.normal(0, 200, n)

    return pd.DataFrame({
        "date": dates, "temp_c": temp.round(1), "rain_mm": rain.round(1),
        "is_holiday": hol, "is_weekend": weekend,
        "park_visitors": park.round().astype(int),
        "hotel_occupancy": np.clip(occ, 20, 100).round(1),
        "rail_riders": rail.round().astype(int),
        "bus_riders": bus.round().astype(int),
        "special_event": event,
    })


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df)} rows to {DATA_PATH}")
