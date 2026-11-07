from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "daily_ops.csv"
REPORT_DIR = ROOT / "reports"

# agg: how a daily series is aggregated into weekly KPIs
METRICS = {
    "park_visitors": dict(label_en="Amusement park visitors", label_ja="遊園地 来園者数", unit="people", agg="sum"),
    "hotel_occupancy": dict(label_en="Hotel occupancy", label_ja="ホテル稼働率", unit="%", agg="mean"),
    "rail_riders": dict(label_en="Railway riders", label_ja="鉄道 乗車人数", unit="people", agg="sum"),
    "bus_riders": dict(label_en="Bus riders", label_ja="バス 乗車人数", unit="people", agg="sum"),
}
