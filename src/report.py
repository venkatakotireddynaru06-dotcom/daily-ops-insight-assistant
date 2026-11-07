"""Automated weekly report: replaces the manual copy-paste-into-Excel routine.

    python -m src.report --week-ending 2025-12-28 --lang ja
"""
from __future__ import annotations
import argparse
import io
import pandas as pd
from .analysis import OpsAnalyzer, load_data
from .config import METRICS, REPORT_DIR

S = {
    "en": dict(title="Weekly Operations Report", period="Period", kpi="Key metrics", metric="Metric",
               this="This week", prev="Previous week", chg="Change %", best="Best day", worst="Lowest day",
               anom="Unusual days", none="No unusual days this week.", fc="7-day outlook (weather = seasonal average)",
               note="Auto-generated from synthetic demo data.", label="label_en"),
    "ja": dict(title="週次オペレーションレポート", period="対象期間", kpi="主要指標", metric="指標",
               this="今週", prev="前週", chg="増減(%)", best="最高日", worst="最低日",
               anom="通常と異なる日", none="今週は異常日なし。", fc="今後7日間の見通し(天候は平年値)",
               note="合成デモデータから自動生成。", label="label_ja"),
}


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)) + "\n"


def build_weekly_report(an: OpsAnalyzer, week_end=None, lang: str = "en") -> dict:
    s = S[lang]
    end = pd.Timestamp(week_end or an.df.date.max())
    start = end - pd.Timedelta(days=6)
    rows, anoms, fc = [], [], None
    for t, meta in METRICS.items():
        k = an.kpis(t, start, end)
        rows.append({s["metric"]: meta[s["label"]], s["this"]: f"{k['value']:,.1f}", s["prev"]: f"{k['previous']:,.1f}" if k["previous"] is not None else "-",
                     s["chg"]: k["change_pct"] if k["change_pct"] is not None else "-",
                     s["best"]: f"{k['best_day']} ({k['best_value']:,.0f})", s["worst"]: f"{k['worst_day']} ({k['worst_value']:,.0f})"})
        a = an.anomalies(t, start=start, end=end)
        if len(a):
            a.insert(0, s["metric"], meta[s["label"]])
            anoms.append(a)
        f = an.forecast(t, 7)[["date", "forecast"]].rename(columns={"forecast": meta[s["label"]]})
        fc = f if fc is None else fc.merge(f, on="date")
    kpi = pd.DataFrame(rows)
    anomalies = pd.concat(anoms) if anoms else pd.DataFrame()
    fc["date"] = fc["date"].dt.strftime("%Y-%m-%d (%a)")
    fc = fc.rename(columns={"date": "日付" if lang == "ja" else "Date"})
    fc = fc.round(0).astype({c: int for c in fc.columns[1:]})
    daily = an.df[(an.df.date >= start) & (an.df.date <= end)][["date", "temp_c", "rain_mm", *METRICS]]

    md = f"# {s['title']}\n\n**{s['period']}:** {start.date()} - {end.date()}\n\n## {s['kpi']}\n\n{_md_table(kpi)}\n## {s['anom']}\n\n"
    md += (_md_table(anomalies.assign(date=anomalies.date.dt.strftime('%Y-%m-%d'))) if len(anomalies) else s["none"] + "\n")
    md += f"\n## {s['fc']}\n\n{_md_table(fc)}\n_{s['note']}_\n"
    return dict(markdown=md, kpi=kpi, daily=daily, anomalies=anomalies, forecast=fc, start=start, end=end)


def to_excel_bytes(res: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name in ("kpi", "daily", "anomalies", "forecast"):
            res[name].to_excel(xw, sheet_name=name.upper(), index=False)
            ws = xw.sheets[name.upper()]
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = max(len(str(c.value or "")) for c in col) + 3
    return buf.getvalue()


def save_report(res: dict, outdir=REPORT_DIR, lang="en"):
    outdir.mkdir(exist_ok=True)
    stem = f"weekly_report_{res['end'].date()}_{lang}"
    (outdir / f"{stem}.md").write_text(res["markdown"], encoding="utf-8")
    (outdir / f"{stem}.xlsx").write_bytes(to_excel_bytes(res))
    return outdir / f"{stem}.md", outdir / f"{stem}.xlsx"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--week-ending")
    p.add_argument("--lang", choices=["en", "ja"], default="en")
    a = p.parse_args()
    r = build_weekly_report(OpsAnalyzer(load_data()), a.week_ending, a.lang)
    print(*save_report(r, lang=a.lang), sep="\n")
