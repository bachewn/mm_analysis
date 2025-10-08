# boxscore_core.py
import pandas as pd
import re

# light canonicalizer — mirrors betting_core.canonicalize_team
def canonicalize_team(name: str) -> str:
    if pd.isna(name):
        return name
    s = str(name).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\b(no\.?|#)\s*\d+\b", "", s, flags=re.IGNORECASE)   # No. 5 / #12
    s = re.sub(r"\bseed\s*\d+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(university|univ|college)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[\.']", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_box_scores(path: str) -> pd.DataFrame:
    """
    Expect columns:
      Year, TeamName, Q1, Q2, Q3, Q4, OT (optional/blank), Total
    """
    df = pd.read_excel(path, sheet_name=0)
    cols_needed = ["Year", "TeamName", "Total"]
    for c in cols_needed:
        if c not in df.columns:
            raise ValueError(f"box scores missing required column: {c}")

    # normalize
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["TeamName"] = df["TeamName"].map(canonicalize_team)

    # ensure numeric for periods if present
    for c in ["Q1", "Q2", "Q3", "Q4", "OT", "Total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep only known columns
    keep = ["Year", "TeamName"] + [c for c in ["Q1","Q2","Q3","Q4","OT","Total"] if c in df.columns]
    return df[keep]

def attach_boxscores_to_games(games_df: pd.DataFrame, box_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join team and opponent quarter scores by (Year, TeamName).
    Produces columns: Team_Q1..Team_Q4..Team_OT..Team_Total and Opp_* equivalents,
    plus per-quarter diffs: Q1_Diff..Q4_Diff..OT_Diff (Team - Opp).
    """
    g = games_df.copy()
    g["Year"] = pd.to_numeric(g["Year"], errors="coerce").astype("Int64")

    # Team-side merge
    team_cols = ["Year","TeamName"] + [c for c in ["Q1","Q2","Q3","Q4","OT","Total"] if c in box_df.columns]
    bt = box_df[team_cols].rename(columns={
        "Q1":"Team_Q1","Q2":"Team_Q2","Q3":"Team_Q3","Q4":"Team_Q4","OT":"Team_OT","Total":"Team_Total"
    })
    g = g.merge(bt, on=["Year","TeamName"], how="left")

    # Opp-side merge
    opp_cols = ["Year","TeamName"] + [c for c in ["Q1","Q2","Q3","Q4","OT","Total"] if c in box_df.columns]
    bo = box_df[opp_cols].rename(columns={
        "TeamName":"OpponentName",
        "Q1":"Opp_Q1","Q2":"Opp_Q2","Q3":"Opp_Q3","Q4":"Opp_Q4","OT":"Opp_OT","Total":"Opp_Total"
    })
    g = g.merge(bo, on=["Year","OpponentName"], how="left")

    # Diffs
    for p in ["Q1","Q2","Q3","Q4","OT"]:
        t = f"Team_{p}"
        o = f"Opp_{p}"
        if t in g.columns and o in g.columns:
            g[f"{p}_Diff"] = g[t] - g[o]

    return g
