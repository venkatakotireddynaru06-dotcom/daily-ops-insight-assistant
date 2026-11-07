# Requirements Definition: Daily Ops Insight Assistant

## 1. Background and problem
Frontline and back-office staff in transport, park and hotel operations spend time every week pulling numbers
into spreadsheets, and have no easy way to anticipate demand or ask "why was Saturday so busy?".
(Modeled on the JD's examples: passenger usage reporting, visitor data, reservation/sales reporting.)

> **To validate:** every statement above is an assumption until checked in a frontline interview.
> See `docs/interview_guide.md` for the question list.

## 2. Users
| User | Need |
|---|---|
| Operations manager (railway/bus) | Weekly ridership summary without manual aggregation |
| Park/hotel planner | 7-day demand outlook to plan staffing |
| Planning dept. (DX team) | Evidence on whether the pilot saves time and is trusted |

## 3. Goals and success metrics (set before the pilot)
| Goal | Metric | Target | Measured in |
|---|---|---|---|
| Cut reporting effort | Minutes per weekly report | -80% | `evaluate.py` + user timing |
| Useful demand outlook | MAPE vs same-weekday baseline | >=30% better | `evaluate.py` |
| Surface unusual days | Recall of known events | >=70% | `tests/` |
| Trust and adoption | User feedback (1-5), weekly usage | >=4 / used weekly | Pilot survey |

## 4. Functional requirements
| ID | Requirement |
|---|---|
| FR1 | Ingest daily metrics for 4 business lines plus weather and holiday calendar from CSV |
| FR2 | Produce a 7-day forecast with an 80% range per metric |
| FR3 | Flag days that deviate from calendar/weather expectation |
| FR4 | Generate a weekly report (Markdown + Excel), English or Japanese, with one command or one click |
| FR5 | Answer natural-language questions (JA/EN) using only computed data |
| FR6 | Show forecast accuracy transparently in the UI |

## 5. Non-functional requirements
- **Security/privacy:** aggregate daily data only; no personal data (no facial or ticket-holder data); API key via environment variable; production would need access control and an approved LLM vendor.
- **Reliability:** the agent never produces numbers itself (all figures from tools); works offline without an API key.
- **Maintainability:** modular code, tests, a one-command setup so a non-developer colleague can run it.
- **Explainability:** unexplained gaps are labeled as such and referred to frontline staff, not "explained" by guesswork.

## 6. Scope
**In:** pilot on one report and one forecast per business line, synthetic data.
**Out (next steps):** live data connectors, real weather-forecast feed, user authentication, hourly granularity, per-ride/per-room models.

## 7. Risks
| Risk | Mitigation |
|---|---|
| Real data is messier than synthetic | Add a validation step and data-quality report before modeling |
| LLM hallucination | Tool-only answers, flagged uncertainty, offline fallback |
| Low adoption | Co-design with frontline users; start from their existing Excel layout |
| Data confidentiality | Aggregates only; vendor review before any cloud LLM use |

## 8. Pilot plan
1. Interview 2-3 frontline users (30 min each) and shadow one reporting session.
2. Replace synthetic data with one real, anonymized aggregate export.
3. Run two weeks in parallel with the manual report; compare numbers and time.
4. Collect feedback, review metrics above, decide: scale, adjust or stop.
