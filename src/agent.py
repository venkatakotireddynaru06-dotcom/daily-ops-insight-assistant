"""Q&A agent over the ops data.

Mode 1 (ANTHROPIC_API_KEY set): Claude with tool use. It can only answer from tool results.
Mode 2 (no key): deterministic offline fallback that routes the question to the same tools.
Either way the numbers come from the analysis layer, never from the model's memory.
"""
from __future__ import annotations
import json
import os
import re
import pandas as pd
from .analysis import OpsAnalyzer
from .config import METRICS

MODEL = os.getenv("OPS_AGENT_MODEL", "claude-sonnet-5-5")
METRIC_ENUM = list(METRICS)

SYSTEM = """You are an operations analyst assistant for a tourism and transport group (railway, bus,
amusement park, hotel). Answer ONLY from tool results; never invent numbers. Reply in the user's language
(Japanese or English), concisely, for a busy frontline manager: lead with the answer, then 1-3 supporting facts.
If a day's result is flagged unexplained, say the data cannot explain it and suggest asking frontline staff
(events, campaigns, incidents). State that forecasts use seasonal-average weather. The data is synthetic demo data."""

TOOLS = [
    dict(name="get_kpis", description="Total/average of a metric over a date range with change vs the previous equal-length period, plus best and worst day.",
         input_schema=dict(type="object", properties=dict(metric=dict(type="string", enum=METRIC_ENUM), start=dict(type="string", description="YYYY-MM-DD"), end=dict(type="string", description="YYYY-MM-DD")), required=["metric", "start", "end"])),
    dict(name="explain_day", description="Explain why one day was higher/lower than expected: actual vs expected, calendar/weather context, whether the gap is unexplained.",
         input_schema=dict(type="object", properties=dict(metric=dict(type="string", enum=METRIC_ENUM), date=dict(type="string", description="YYYY-MM-DD")), required=["metric", "date"])),
    dict(name="forecast", description="Forecast of a metric for the next N days with an 80% range.",
         input_schema=dict(type="object", properties=dict(metric=dict(type="string", enum=METRIC_ENUM), days=dict(type="integer", minimum=1, maximum=14)), required=["metric"])),
    dict(name="find_anomalies", description="List unusual days for a metric in a date range.",
         input_schema=dict(type="object", properties=dict(metric=dict(type="string", enum=METRIC_ENUM), start=dict(type="string"), end=dict(type="string")), required=["metric"])),
]


def _records(df: pd.DataFrame) -> list:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records"))


class OpsTools:
    def __init__(self, an: OpsAnalyzer):
        self.an = an

    def run(self, name: str, a: dict) -> dict:
        try:
            if name == "get_kpis":
                return self.an.kpis(a["metric"], a["start"], a["end"])
            if name == "explain_day":
                return self.an.explain_day(a["date"], a["metric"])
            if name == "forecast":
                return dict(metric=a["metric"], rows=_records(self.an.forecast(a["metric"], int(a.get("days", 7)))))
            if name == "find_anomalies":
                end = a.get("end") or self.an.df.date.max()
                start = a.get("start") or pd.Timestamp(end) - pd.Timedelta(days=90)
                return dict(metric=a["metric"], rows=_records(self.an.anomalies(a["metric"], start=start, end=end)))
            return dict(error=f"unknown tool {name}")
        except Exception as e:  # surface errors to the model instead of crashing
            return dict(error=str(e))


class OpsAgent:
    def __init__(self, an: OpsAnalyzer):
        self.an, self.tools = an, OpsTools(an)
        self.client = None
        if os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            self.client = anthropic.Anthropic()

    @property
    def mode(self) -> str:
        return f"Claude API ({MODEL})" if self.client else "offline fallback (set ANTHROPIC_API_KEY for the LLM agent)"

    def answer(self, question: str) -> str:
        return self._llm(question) if self.client else self._offline(question)

    # ---- LLM tool-use loop ----------------------------------------------------------
    def _llm(self, question: str) -> str:
        last = self.an.df.date.max().date()
        msgs = [dict(role="user", content=f"(Data runs through {last}.)\n{question}")]
        for _ in range(6):
            r = self.client.messages.create(model=MODEL, max_tokens=1000, system=SYSTEM, tools=TOOLS, messages=msgs)
            if r.stop_reason != "tool_use":
                return "".join(b.text for b in r.content if b.type == "text")
            msgs.append(dict(role="assistant", content=r.content))
            results = [dict(type="tool_result", tool_use_id=b.id, content=json.dumps(self.tools.run(b.name, b.input), default=str))
                       for b in r.content if b.type == "tool_use"]
            msgs.append(dict(role="user", content=results))
        return "Sorry, I could not finish the analysis."

    @staticmethod
    def _resolve_date(ql: str, end: pd.Timestamp):
        """Turn 'last saturday', 'yesterday', '土曜' or YYYY-MM-DD into a date string (relative to the data end)."""
        m = re.search(r"\d{4}-\d{2}-\d{2}", ql)
        if m:
            return m.group(0)
        if re.search(r"yesterday|昨日", ql):
            return str((end - pd.Timedelta(days=1)).date())
        if re.search(r"\btoday\b|今日", ql):
            return str(end.date())
        en = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
        wd = next((i for n, i in en.items() if re.search(r"\b" + n + r"\b", ql)), None)
        j = re.search(r"([月火水木金土日])曜", ql)
        if wd is None and j:
            wd = "月火水木金土日".index(j.group(1))
        if wd is None:
            return None
        return str((end - pd.Timedelta(days=(end.dayofweek - wd) % 7)).date())

    # ---- offline fallback -----------------------------------------------------------
    def _offline(self, q: str) -> str:
        ja = bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", q))
        ql = q.lower()
        kw = {"hotel_occupancy": ["hotel", "occupancy", "ホテル", "稼働"], "rail_riders": ["rail", "railway", "train", "電車", "鉄道", "乗車"],
              "bus_riders": ["bus", "バス"]}
        metric = next((m for m, ws in kw.items() if any(re.search(r"\b" + w + r"\b" if w.isascii() else w, ql) for w in ws)), "park_visitors")
        label = METRICS[metric]["label_ja" if ja else "label_en"]
        end = self.an.df.date.max()
        date = self._resolve_date(ql, end)

        if date or re.search(r"\bwhy\b|なぜ|なんで|busier|\bbusy\b|多かった|少なかった", ql):
            if not date:
                a = self.an.anomalies(metric, start=end - pd.Timedelta(days=90))
                date = str((a.iloc[-1].date if len(a) else end).date())
            r = self.tools.run("explain_day", dict(metric=metric, date=date))
            if "error" in r:
                return r["error"]
            ctx = ", ".join(r["context"]) or ("特になし" if ja else "none")
            if ja:
                tail = "カレンダー・天候では説明できない差です。イベントやキャンペーンなど現場への確認をおすすめします。" if r["unexplained"] else "カレンダーと天候から見て想定どおりの結果です。"
                return f"{r['date']} の{label}は {r['actual']:,.0f}(想定 {r['expected']:,.0f}、差 {r['diff_vs_expected_pct']:+.1f}%)。背景: {ctx}。{tail}"
            tail = "This gap isn't explained by calendar or weather; check with frontline staff for events or campaigns." if r["unexplained"] else "This is in line with what calendar and weather predict."
            return f"{label} on {r['date']}: {r['actual']:,.0f} vs {r['expected']:,.0f} expected ({r['diff_vs_expected_pct']:+.1f}%). Context: {ctx}. {tail}"

        if re.search(r"forecast|predict|next|予測|見通し|来週", ql):
            rows = self.tools.run("forecast", dict(metric=metric, days=7))["rows"]
            lines = "\n".join(f"- {r['date']}: {r['forecast']:,.0f} ({r['low']:,.0f}-{r['high']:,.0f})" for r in rows)
            return (f"{label}の今後7日間予測(天候は平年値):\n" if ja else f"{label} forecast, next 7 days (weather = seasonal average):\n") + lines

        if re.search(r"anomal|unusual|spike|異常|急増|急減", ql):
            rows = self.tools.run("find_anomalies", dict(metric=metric))["rows"]
            if not rows:
                return "直近90日に異常日はありません。" if ja else "No unusual days in the last 90 days."
            lines = "\n".join(f"- {r['date']}: {r['actual']:,.0f} (expected {r['expected']:,.0f}, {r['diff_pct']:+.1f}%)" for r in rows)
            return ("直近90日の異常日:\n" if ja else "Unusual days in the last 90 days:\n") + lines

        k = self.tools.run("get_kpis", dict(metric=metric, start=str(end - pd.Timedelta(days=6)), end=str(end)))
        chg = "n/a" if k["change_pct"] is None else f"{k['change_pct']:+.1f}%"
        if ja:
            return f"直近7日間({k['start']}〜{k['end']})の{label}: {k['value']:,.1f}(前週比 {chg})。最高日 {k['best_day']}、最低日 {k['worst_day']}。"
        return f"{label}, last 7 days ({k['start']} to {k['end']}): {k['value']:,.1f} ({chg} vs prior week). Best day {k['best_day']}, lowest {k['worst_day']}."
