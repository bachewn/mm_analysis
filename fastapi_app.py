
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import pandas as pd
from mm_core import compute_win_pct, compute_filtered_df, load_dataset, compute_lower_seed_win_pct

DATA_PATH = os.getenv("MM_DATA_PATH", "./df_first_round_all_years.xlsx")
app = FastAPI(title="MM First-Round API")
_df = load_dataset(DATA_PATH)

class WinPctRequest(BaseModel):
    pace_diff_max: float = 2.0
    require_higher_seed: bool = True
    require_positive_net: bool = True
    by_year: bool = True
    years: Optional[List[int]] = None
    counts: bool = True

@app.post("/win_pct")
def win_pct(req: WinPctRequest):
    if req.counts:
        result = compute_win_pct(
            _df,
            pace_diff_max=req.pace_diff_max,
            require_higher_seed=req.require_higher_seed,
            require_positive_net=req.require_positive_net,
            by_year=req.by_year,
            years=req.years,
            return_counts=True,
        )
        return result
    else:
        result = compute_win_pct(
            _df,
            pace_diff_max=req.pace_diff_max,
            require_higher_seed=req.require_higher_seed,
            require_positive_net=req.require_positive_net,
            by_year=req.by_year,
            years=req.years,
        )
        if req.by_year:
            return {"win_pct_by_year": result.to_dict()}
        else:
            return {"win_pct_overall": float(result)}

class LowerSeedRequest(BaseModel):
    pace_diff_max: float = 2.0
    by_year: bool = True
    years: Optional[List[int]] = None
    counts: bool = True

@app.post("/lower_seed_win_pct")
def lower_seed_win_pct(req: LowerSeedRequest):
    if req.counts:
        result = compute_lower_seed_win_pct(
            _df,
            pace_diff_max=req.pace_diff_max,
            by_year=req.by_year,
            years=req.years,
            return_counts=True,
        )
        return result
    else:
        result = compute_lower_seed_win_pct(
            _df,
            pace_diff_max=req.pace_diff_max,
            by_year=req.by_year,
            years=req.years,
        )
        if req.by_year:
            return {"win_pct_by_year": result.to_dict()}
        else:
            return {"win_pct_overall": float(result)}
