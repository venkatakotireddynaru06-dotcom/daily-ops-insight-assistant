# Pilot Evaluation Results (auto-generated)

Data: synthetic, 1096 days. Forecast test window: last 90 days (never seen in training).

## 1. Forecast accuracy (MAPE, lower is better)
| Metric | Baseline: same weekday last week | Model | Improvement |
|---|---|---|---|
| Amusement park visitors | 20.32% | 7.0% | 66% |
| Hotel occupancy | 7.75% | 5.0% | 35% |
| Railway riders | 8.8% | 4.59% | 48% |
| Bus riders | 7.08% | 4.53% | 36% |

## 2. Anomaly detection (park visitors, validated against hidden event days)
- Hidden event days: 24 | flagged days: 24
- Recall: 83% | Precision: 83%

## 3. Report automation
- Automated weekly report generation (pipeline + analysis + Excel/Markdown): 5.6 s
- Assumed manual effort: 45 min/week (**assumption: replace with real interview data**)
- Estimated time saved: ~45 min/week per report owner (~39 h/year)

## 4. Caveats
- Synthetic data: accuracy numbers show the method works, not what real operations would achieve.
- Forecast uses seasonal-average weather; real use would plug in a weather forecast feed.
