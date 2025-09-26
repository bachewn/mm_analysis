
# March Madness First-Round — Starter Kit

This folder contains a **core analysis module**, a **CLI**, and a **minimal FastAPI API** for your first metric:
> Win % when |Team_Pace - Opp_Pace| ≤ 2 and the higher-seeded team's NET_Diff > 0

## Files
- `mm_core.py` — reusable analysis functions
- `cli.py` — Typer CLI to run the metric locally
- `fastapi_app.py` — FastAPI endpoints exposing the metric
- `df_first_round_all_years.xlsx` — your input data (2021–2025)

## How to run the CLI
```bash
python cli.py winpct --path "/mnt/data/df_first_round_all_years.xlsx" --pace-diff-max 2 --require-higher-seed True --require-positive-net True --by-year True --export-csv /mnt/data/filtered_rows.csv
```

## How to run the API (locally)
```bash
uvicorn fastapi_app:app --reload --port 8000
```
Then POST to `http://127.0.0.1:8000/win_pct` with JSON:
```json
{"pace_diff_max": 2, "require_higher_seed": true, "require_positive_net": true, "by_year": true}
```

## Next steps
- Swap Excel for Parquet (fast) or Postgres (scalable).
- Add more metrics as new functions in `mm_core.py` and expose via CLI + API.
- Wrap with a simple Streamlit/Next.js frontend to let users tweak filters and see charts.
