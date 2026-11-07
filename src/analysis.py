"""Forecasting, anomaly detection and day-level explanations."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from .calendar_jp import is_holiday
from .config import DATA_PATH, METRICS

FEATURES = ["dow", "month", "doy_sin", "doy_cos", "is_holiday", "is_weekend", "temp_c", "rain_mm"]


def load_data(path=DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        from .generate_data import generate
        Path(path).parent.mkdir(exist_ok=True)
        generate().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dow"], d["month"] = d.date.dt.dayofweek, d.date.dt.month
    doy = d.date.dt.dayofyear
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["is_weekend"] = (d.dow >= 5).astype(int)
    return d


def _model():
    return GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3,
                                     subsample=0.8, random_state=0)


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p) / np.abs(y)) * 100)


class OpsAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = _time_features(df).reset_index(drop=True)
        doy = self.df.date.dt.dayofyear
        clim = self.df.groupby(doy)[["temp_c", "rain_mm"]].mean()
        self.climate = clim.reindex(range(1, 367)).interpolate().bfill().ffill()
        self._oof: dict[str, pd.Series] = {}

    # ---- model-based expectations -------------------------------------------------
    def oof(self, target: str) -> pd.Series:
        """Out-of-fold expected value per day (the model never sees the day it predicts)."""
        if target not in self._oof:
            pred = cross_val_predict(_model(), self.df[FEATURES], self.df[target], cv=KFold(6))
            self._oof[target] = pd.Series(pred, index=self.df.index)
        return self._oof[target]

    def backtest(self, target: str, test_days: int = 90) -> dict:
        d = self.df
        cut = d.date.max() - pd.Timedelta(days=test_days)
        tr, te = d[d.date <= cut], d[d.date > cut]
        pred = _model().fit(tr[FEATURES], tr[target]).predict(te[FEATURES])
        naive = d[target].shift(7).loc[te.index]  # baseline: same weekday last week
        return dict(metric=target, test_days=test_days,
                    mape_model=round(mape(te[target], pred), 2), mape_naive=round(mape(te[target], naive), 2),
                    mae_model=round(float(np.mean(np.abs(te[target] - pred))), 1),
                    mae_naive=round(float(np.mean(np.abs(te[target] - naive))), 1))

    def forecast(self, target: str, days: int = 7) -> pd.DataFrame:
        d = self.df
        model = _model().fit(d[FEATURES], d[target])
        dates = pd.date_range(d.date.max() + pd.Timedelta(days=1), periods=days)
        f = _time_features(pd.DataFrame({"date": dates}))
        f["is_holiday"] = [int(is_holiday(x)) for x in dates]
        doy = f.date.dt.dayofyear
        f["temp_c"], f["rain_mm"] = self.climate.loc[doy, "temp_c"].values, self.climate.loc[doy, "rain_mm"].values
        pred = model.predict(f[FEATURES])
        lo, hi = (d[target] - self.oof(target)).quantile([0.1, 0.9])
        f["forecast"], f["low"], f["high"] = pred, pred + lo, pred + hi
        if target == "hotel_occupancy":
            f[["forecast", "low", "high"]] = f[["forecast", "low", "high"]].clip(0, 100)
        out = f[["date", "forecast", "low", "high", "is_holiday", "temp_c", "rain_mm"]].copy()
        out[["forecast", "low", "high", "temp_c", "rain_mm"]] = out[["forecast", "low", "high", "temp_c", "rain_mm"]].round(1)
        return out

    # ---- anomalies & explanations -------------------------------------------------
    def anomalies(self, target: str, z: float = 2.5, start=None, end=None) -> pd.DataFrame:
        d, exp = self.df, self.oof(target)
        pct = (d[target] - exp) / exp * 100
        zs = (pct - pct.mean()) / pct.std()
        out = pd.DataFrame({"date": d.date, "actual": d[target], "expected": exp.round(1),
                            "diff_pct": pct.round(1), "z": zs.round(2)})
        out = out[out.z.abs() >= z]
        if start is not None:
            out = out[out.date >= pd.Timestamp(start)]
        if end is not None:
            out = out[out.date <= pd.Timestamp(end)]
        return out.reset_index(drop=True)

    def explain_day(self, date, target: str) -> dict:
        date = pd.Timestamp(date)
        row = self.df[self.df.date == date]
        if row.empty:
            raise ValueError(f"No data for {date.date()}")
        i, r = row.index[0], row.iloc[0]
        exp, act = float(self.oof(target)[i]), float(r[target])
        diff = (act - exp) / exp * 100
        clim = self.climate.loc[date.dayofyear]
        drivers = []
        if r.is_holiday: drivers.append("public holiday")
        if r.is_weekend: drivers.append("weekend")
        if r.rain_mm >= 10: drivers.append(f"rain ({r.rain_mm:.0f} mm)")
        if r.temp_c - clim.temp_c >= 5: drivers.append(f"unusually warm ({r.temp_c:.0f}C)")
        if r.temp_c - clim.temp_c <= -5: drivers.append(f"unusually cold ({r.temp_c:.0f}C)")
        same = self.df[(self.df.dow == r.dow) & (self.df.date < date) & (self.df.date >= date - pd.Timedelta(weeks=8))][target]
        vs_same = float((act - same.mean()) / same.mean() * 100) if len(same) else None
        unexplained = abs(diff) >= 15
        note = ("Large gap vs. the level expected from calendar and weather: likely an event, campaign or "
                "operational factor. Confirm with frontline staff." if unexplained
                else "Result is in line with what calendar and weather would predict.")
        return dict(date=str(date.date()), metric=target, actual=round(act, 1), expected=round(exp, 1),
                    diff_vs_expected_pct=round(diff, 1),
                    diff_vs_same_weekday_avg_pct=None if vs_same is None else round(vs_same, 1),
                    context=drivers, unexplained=bool(unexplained), note=note)

    def kpis(self, target: str, start, end) -> dict:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        n = (end - start).days + 1
        d = self.df
        cur = d[(d.date >= start) & (d.date <= end)]
        prev = d[(d.date >= start - pd.Timedelta(days=n)) & (d.date < start)]
        if cur.empty:
            raise ValueError("No data in range")
        agg = (lambda s: s.sum()) if METRICS[target]["agg"] == "sum" else (lambda s: s.mean())
        c = float(agg(cur[target]))
        p = float(agg(prev[target])) if len(prev) else None
        best, worst = cur.loc[cur[target].idxmax()], cur.loc[cur[target].idxmin()]
        return dict(metric=target, start=str(start.date()), end=str(end.date()), value=round(c, 1),
                    previous=None if p is None else round(p, 1),
                    change_pct=None if not p else round((c - p) / p * 100, 1),
                    best_day=str(best.date.date()), best_value=float(best[target]),
                    worst_day=str(worst.date.date()), worst_value=float(worst[target]))
