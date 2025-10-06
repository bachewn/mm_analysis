# analyze_march_madness_boxscores.py
"""
Analyze first-round NCAA game box scores vs outcomes.

Requires:
    pip install pandas typer[all] openpyxl

Usage examples:
    # Basic: win rate when team leads by >=6 at end of Q1
    python analyze_march_madness_boxscores.py condition --qstage Q1 --op ge --lead 6

    # Win rate when team trails by <= -4 at halftime (i.e., opponent leads by 4+)
    python analyze_march_madness_boxscores.py condition --qstage HALF --op le --lead -4

    # Between +3 and +8 at end of Q3
    python analyze_march_madness_boxscores.py condition --qstage Q3 --op between --lead -3 --lead_hi 8

    # Scan win rate by lead bins for HALF, from -20..20 in steps of 2, and save CSV
    python analyze_march_madness_boxscores.py scan --qstage HALF --lo -20 --hi 20 --step 2 --out scan_half.csv

Inputs:
  - df_first_round_all_years.xlsx  (columns include Year, TeamName, OpponentName, Team_Won, etc.)
  - box_scores.xlsx                (columns: Year, TeamName, Q1,Q2,Q3,Q4,OT,Total)
"""

import pandas as pd
import typer
from typing import Optional

app = typer.Typer(add_completion=False)

def _load_data(
    path_map: str = r"C:\Users\888\Desktop\2024 NCAA Stats\df_first_round_all_years.xlsx",
    path_scores: str = r"C:\Users\888\Desktop\2024 NCAA Stats\box_scores.xlsx",
) -> pd.DataFrame:
    # Load both files
    dfm = pd.read_excel(path_map)
    dfs = pd.read_excel(path_scores)

    # --- Merge prep ---
    for c in ["Q1", "Q2", "Q3", "Q4", "OT", "Total"]:
        dfs[c] = pd.to_numeric(dfs[c], errors="coerce").fillna(0)

    # --- Perform the merges (use prefixed keys!) ---
    merged = dfm.merge(
        dfs.add_prefix("Team_"),
        left_on=["Year", "TeamName"],
        right_on=["Team_Year", "Team_TeamName"],  # <— was ["Year","Team_TeamName"]
        how="left",
    )

    merged = merged.merge(
        dfs.add_prefix("Opp_"),
        left_on=["Year", "OpponentName"],
        right_on=["Opp_Year", "Opp_TeamName"],    # <— was ["Year","Opp_TeamName"]
        how="left",
        suffixes=("", "_oppdupe"),
    )

    # --- Minimal merge-audit: print names that didn't match ---
    missing_team = merged[merged["Team_TeamName"].isna()][["Year", "TeamName"]].drop_duplicates().sort_values(["Year","TeamName"])
    missing_opp  = merged[merged["Opp_TeamName"].isna()][["Year", "OpponentName"]].drop_duplicates().sort_values(["Year","OpponentName"])

    if not missing_team.empty or not missing_opp.empty:
        print("\n⚠️  Merge issues detected.")
        if not missing_team.empty:
            print("\nTeams in mapping not found in box_scores (TeamName side):")
            print(missing_team.to_string(index=False))
        if not missing_opp.empty:
            print("\nTeams in mapping not found in box_scores (OpponentName side):")
            print(missing_opp.to_string(index=False))
    else:
        print("\n✅ All team names merged successfully!")


    # --- Continue normal calculations if desired ---
    merged["Team_cum_Q1"] = merged["Team_Q1"]
    merged["Team_cum_HALF"] = merged["Team_Q1"] + merged["Team_Q2"]
    merged["Team_cum_Q3"] = merged["Team_Q1"] + merged["Team_Q2"] + merged["Team_Q3"]
    merged["Team_cum_REG"] = merged["Team_Q1"] + merged["Team_Q2"] + merged["Team_Q3"] + merged["Team_Q4"]
    merged["Team_cum_FINAL"] = merged["Team_cum_REG"] + merged["Team_OT"]

    merged["Opp_cum_Q1"] = merged["Opp_Q1"]
    merged["Opp_cum_HALF"] = merged["Opp_Q1"] + merged["Opp_Q2"]
    merged["Opp_cum_Q3"] = merged["Opp_Q1"] + merged["Opp_Q2"] + merged["Opp_Q3"]
    merged["Opp_cum_REG"] = merged["Opp_Q1"] + merged["Opp_Q2"] + merged["Opp_Q3"] + merged["Opp_Q4"]
    merged["Opp_cum_FINAL"] = merged["Opp_cum_REG"] + merged["Opp_OT"]

    for stage in ["Q1", "HALF", "Q3", "REG", "FINAL"]:
        merged[f"Lead_{stage}"] = merged[f"Team_cum_{stage}"] - merged[f"Opp_cum_{stage}"]

    merged["Team_Won"] = pd.to_numeric(merged["Team_Won"], errors="coerce").fillna(0).astype(int)

    score_cols = ["Team_Q1","Team_Q2","Team_Q3","Team_Q4","Opp_Q1","Opp_Q2","Opp_Q3","Opp_Q4"]
    df_valid = merged.dropna(subset=score_cols)
    return df_valid



def _apply_condition(series: pd.Series, op: str, lead: int, lead_hi: Optional[int] = None) -> pd.Series:
    if op == "ge":
        return series >= lead
    elif op == "gt":
        return series > lead
    elif op == "le":
        return series <= lead
    elif op == "lt":
        return series < lead
    elif op == "eq":
        return series == lead
    elif op == "between":
        if lead_hi is None:
            raise ValueError("For op='between', you must provide --lead_hi.")
        lo, hi = sorted((lead, lead_hi))
        return (series >= lo) & (series <= hi)
    else:
        raise ValueError(f"Unknown op: {op}. Use one of ge,gt,le,lt,eq,between.")


def _qstage_to_col(qstage: str) -> str:
    qstage = qstage.upper()
    mapping = {"Q1": "Lead_Q1", "Q2": "Lead_HALF", "HALF": "Lead_HALF",
               "Q3": "Lead_Q3", "Q4": "Lead_REG", "REG": "Lead_REG",
               "FINAL": "Lead_FINAL"}
    if qstage not in mapping:
        raise ValueError("qstage must be one of: Q1, Q2/HALF, Q3, Q4/REG, FINAL")
    return mapping[qstage]


@app.command()
def condition(
    qstage: str = typer.Option(..., help="Stage to condition on: Q1, HALF (or Q2), Q3, REG (or Q4), FINAL"),
    op: str = typer.Option("ge", help="Comparison: ge, gt, le, lt, eq, between"),
    lead: int = typer.Option(..., help="Lead threshold (Team - Opponent). Use negative for trailing conditions."),
    lead_hi: Optional[int] = typer.Option(None, help="Upper bound if op='between'"),
    map_path: Optional[str] = typer.Option(None, help="Path to mapping Excel"),
    scores_path: Optional[str] = typer.Option(None, help="Path to box scores Excel"),
    year_min: Optional[int] = typer.Option(None, help="Filter: minimum Year (inclusive)"),
    year_max: Optional[int] = typer.Option(None, help="Filter: maximum Year (inclusive)"),
    print_examples: int = typer.Option(5, help="Show up to N example rows matching the condition."),
    out: Optional[str] = typer.Option(None, help="Optional CSV path to save ALL matching rows"),
):

    """
    Compute conditional win rate given a lead condition at a stage
    (e.g., lead >= 6 at end of Q1).
    """
    df = _load_data(
        path_map=map_path or r"C:\Users\888\Desktop\2024 NCAA Stats\df_first_round_all_years.xlsx",
        path_scores=scores_path or r"C:\Users\888\Desktop\2024 NCAA Stats\box_scores.xlsx",
    )
    if year_min is not None:
        df = df[df["Year"] >= year_min]
    if year_max is not None:
        df = df[df["Year"] <= year_max]

    col = _qstage_to_col(qstage)
    mask = _apply_condition(df[col], op, lead, lead_hi)
    subset = df[mask].copy()

    total = len(subset)
    wins = int(subset["Team_Won"].sum())
    win_rate = wins / total if total else float("nan")

    typer.echo(f"Condition: {col} {op} {lead}" + (f" .. {lead_hi}" if op == "between" else ""))
    typer.echo(f"Games matching condition: {total}")
    typer.echo(f"Wins: {wins}")
    typer.echo(f"Win rate: {win_rate:.3f}" if total else "Win rate: N/A (no matches)")

    # Columns to show/keep (print-safe if some are missing)
    want_cols = [
        "Year", "TeamName", "OpponentName", col, "Team_Won",
        "Team_cum_Q1", "Opp_cum_Q1", "Team_cum_HALF", "Opp_cum_HALF",
        "Team_cum_Q3", "Opp_cum_Q3", "Team_cum_REG", "Opp_cum_REG",
        "Team_cum_FINAL", "Opp_cum_FINAL"
    ]
    show_cols = [c for c in want_cols if c in subset.columns]
    subset_print = subset[show_cols].sort_values(["Year", col, "TeamName"], ascending=[False, False, True])

    # Save ALL matches if requested
    if out:
        subset_print.to_csv(out, index=False)
        typer.echo(f"\nSaved all {total} matching rows to {out}")

    # Print matches
    if total:
        typer.echo("\nMatches:")
        if print_examples == -1:
            # print ALL
            typer.echo(subset_print.to_string(index=False))
        elif print_examples > 0:
            typer.echo(subset_print.head(print_examples).to_string(index=False))
        else:
            typer.echo("(printing suppressed; use --print-examples -1 for all or set a positive number)")


@app.command()
def scan(
    qstage: str = typer.Option(..., help="Stage to scan over: Q1, HALF (or Q2), Q3, REG (or Q4), FINAL"),
    lo: int = typer.Option(-20, help="Min lead to scan (inclusive)"),
    hi: int = typer.Option(20, help="Max lead to scan (inclusive)"),
    step: int = typer.Option(2, help="Step size for lead"),
    op: str = typer.Option("ge", help="Per-lead operator: ge, gt, le, lt, eq (ignored if --bands used)"),
    bands: Optional[str] = typer.Option(None, help="Optional comma-separated band edges, e.g. '-20,-10,-5,0,5,10,20' to compute between-bucket win rates."),
    map_path: Optional[str] = typer.Option(None, help="Path to mapping Excel"),
    scores_path: Optional[str] = typer.Option(None, help="Path to box scores Excel"),
    year_min: Optional[int] = typer.Option(None),
    year_max: Optional[int] = typer.Option(None),
    out: Optional[str] = typer.Option(None, help="Optional CSV path to save results"),
):
    """
    Scan win rates across lead thresholds or into custom bands at a stage.

    - Threshold mode (default): iterates lead from lo..hi and applies (lead_col op threshold).
    - Band mode (--bands): computes win rate within each inclusive band [edge_i, edge_{i+1}].
    """
    df = _load_data(
        path_map=map_path or r"C:\Users\888\Desktop\2024 NCAA Stats\df_first_round_all_years.xlsx",
        path_scores=scores_path or r"C:\Users\888\Desktop\2024 NCAA Stats\box_scores.xlsx",
    )
    if year_min is not None:
        df = df[df["Year"] >= year_min]
    if year_max is not None:
        df = df[df["Year"] <= year_max]

    col = _qstage_to_col(qstage)

    results = []
    if bands:
        # Custom bands: e.g., "-20,-10,-5,0,5,10,20"
        edges = [int(x.strip()) for x in bands.split(",")]
        if len(edges) < 2:
            raise ValueError("Provide at least two band edges.")
        edges = sorted(edges)
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (df[col] >= a) & (df[col] <= b)
            total = int(mask.sum())
            wins = int(df.loc[mask, "Team_Won"].sum())
            win_rate = wins / total if total else float("nan")
            results.append({"band_lo": a, "band_hi": b, "games": total, "wins": wins, "win_rate": win_rate})
    else:
        # Threshold scan: for t in [lo, lo+step, ..., hi]
        for t in range(lo, hi + 1, step):
            mask = _apply_condition(df[col], op, t, None)
            total = int(mask.sum())
            wins = int(df.loc[mask, "Team_Won"].sum())
            win_rate = wins / total if total else float("nan")
            results.append({"threshold": t, "op": op, "games": total, "wins": wins, "win_rate": win_rate})

    res = pd.DataFrame(results)
    typer.echo(res.to_string(index=False))

    if out:
        res.to_csv(out, index=False)
        typer.echo(f"\nSaved to {out}")


@app.command()
def describe(
    map_path: Optional[str] = typer.Option(None, help="Path to mapping Excel"),
    scores_path: Optional[str] = typer.Option(None, help="Path to box scores Excel"),
):
    """
    Quick sanity checks and available columns/stages.
    """
    df = _load_data(
        path_map=map_path or r"C:\Users\888\Desktop\2024 NCAA Stats\df_first_round_all_years.xlsx",
        path_scores=scores_path or r"C:\Users\888\Desktop\2024 NCAA Stats\box_scores.xlsx",
    )
    typer.echo(f"Rows (team-rows) after merge: {len(df)}")
    typer.echo("\nStages available: Q1, HALF (Q2 cumulative), Q3, REG (Q4 cumulative), FINAL (with OT)")
    typer.echo("\nPreview:")
    cols = ["Year", "TeamName", "OpponentName", "Team_Won",
            "Lead_Q1", "Lead_HALF", "Lead_Q3", "Lead_REG", "Lead_FINAL"]
    typer.echo(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    app()
