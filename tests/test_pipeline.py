import pandas as pd
import pytest
from src.agent import OpsAgent
from src.analysis import OpsAnalyzer
from src.config import METRICS
from src.generate_data import generate
from src.report import build_weekly_report


@pytest.fixture(scope="module")
def an():
    return OpsAnalyzer(generate())


def test_data_shape():
    df = generate()
    assert len(df) == 1096 and df.isna().sum().sum() == 0
    assert df.hotel_occupancy.between(0, 100).all()


@pytest.mark.parametrize("metric", list(METRICS))
def test_model_beats_baseline(an, metric):
    b = an.backtest(metric)
    assert b["mape_model"] < b["mape_naive"]


def test_anomaly_recall(an):
    df = an.df
    flagged = set(an.anomalies("park_visitors").date)
    events = set(df[df.special_event == 1].date)
    assert len(flagged & events) / len(events) >= 0.7


def test_forecast_shape_and_bounds(an):
    f = an.forecast("hotel_occupancy", 7)
    assert len(f) == 7 and (f.low >= 0).all() and (f.high <= 100).all()
    assert (f.low <= f.forecast).all() and (f.forecast <= f.high).all()


def test_report_has_all_sections(an):
    for lang in ("en", "ja"):
        r = build_weekly_report(an, lang=lang)
        assert len(r["kpi"]) == len(METRICS) and len(r["forecast"]) == 7


def test_offline_agent_routes_and_busy_not_bus(an, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ag = OpsAgent(an)
    assert "Amusement park" in ag.answer("Why was the park so busy recently?") or "park" in ag.answer("Why was the park so busy?").lower()
    assert "forecast" in ag.answer("hotel forecast").lower()
    assert "鉄道" in ag.answer("電車の今週の状況")
