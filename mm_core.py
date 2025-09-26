
import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    """Load the first sheet from the Excel file into a DataFrame."""
    return pd.read_excel(path, sheet_name=0)


def _apply_years(df: pd.DataFrame, years=None) -> pd.DataFrame:
    if years is None:
        return df
    if isinstance(years, (int, float)):
        years = [int(years)]
    years = [int(y) for y in years]
    return df[df["Year"].isin(years)]

def compute_win_pct(
    df: pd.DataFrame,
    pace_diff_max: float = 2.0,
    require_higher_seed: bool = True,
    require_positive_net: bool = True,
    by_year: bool = False,
    years=None,
    return_counts: bool = False,
):
    """
    Win % for (optionally filtered) years, with optional counts.
    Returns float/Series by default; if return_counts=True returns a dict with n, wins, pct, and (optionally) year breakdowns.
    """
    df = _apply_years(df.copy(), years)
    df["Pace_Diff"] = (df["Team_Pace"] - df["Opp_Pace"]).abs()
    mask = df["Pace_Diff"] <= pace_diff_max

    if require_higher_seed:
        is_higher = df["Team_Seed"] < df["Opp_Seed"]
        mask &= is_higher
        if require_positive_net:
            mask &= df["NET_Diff"] > 0
    else:
        if require_positive_net:
            mask &= df["NET_Diff"] > 0

    filtered = df[mask]
    if return_counts:
        n = int(len(filtered))
        wins = int(filtered["Team_Won"].sum())
        pct = float((wins / n) * 100.0) if n else 0.0
        if by_year:
            group = filtered.groupby("Year")["Team_Won"]
            by_year_counts = (
                filtered.groupby("Year")
                .agg(n=("Team_Won", "size"), wins=("Team_Won", "sum"))
                .assign(pct=lambda d: (d["wins"] / d["n"] * 100.0).round(6))
            )
            return {
                "overall": {"n": n, "wins": wins, "pct": pct},
                "by_year": by_year_counts.reset_index().to_dict(orient="records"),
            }
        return {"overall": {"n": n, "wins": wins, "pct": pct}}

    if by_year:
        return filtered.groupby("Year")["Team_Won"].mean() * 100.0
    else:
        return float(filtered["Team_Won"].mean() * 100.0)


def compute_lower_seed_win_pct(
    df: pd.DataFrame,
    pace_diff_max: float = 2.0,
    by_year: bool = False,
    years=None,
    return_counts: bool = False,
    require_higher_seed_net_negative: bool = True,  # NEW: toggle NET condition
):
    """
    Percentage of lower-seeded teams (numerically higher seed) that win.

    Conditions:
      - abs(Team_Pace - Opp_Pace) <= pace_diff_max
      - Row is lower-seeded (Team_Seed > Opp_Seed)
      - If require_higher_seed_net_negative=True:
          Higher-seeded team's NET_Diff < 0  ==> on the lower-seeded row NET_Diff > 0
    """
    df = _apply_years(df.copy(), years)
    df["Pace_Diff"] = (df["Team_Pace"] - df["Opp_Pace"]).abs()
    is_lower = df["Team_Seed"] > df["Opp_Seed"]

    cond = (df["Pace_Diff"] <= pace_diff_max) & is_lower
    if require_higher_seed_net_negative:
        cond = cond & (df["NET_Diff"] > 0)

    filt = df[cond]

    if return_counts:
        n = int(len(filt))
        wins = int(filt["Team_Won"].sum())
        pct = float((wins / n) * 100.0) if n else 0.0
        if by_year:
            by_year_counts = (
                filt.groupby("Year")
                .agg(n=("Team_Won", "size"), wins=("Team_Won", "sum"))
                .assign(pct=lambda d: (d["wins"] / d["n"] * 100.0).round(6))
                .reset_index()
            )
            return {"overall": {"n": n, "wins": wins, "pct": pct},
                    "by_year": by_year_counts.to_dict(orient="records")}
        return {"overall": {"n": n, "wins": wins, "pct": pct}}

    if by_year:
        return filt.groupby("Year")["Team_Won"].mean() * 100.0
    return float(filt["Team_Won"].mean() * 100.0)


def compute_lower_seed_filtered_df(
    df: pd.DataFrame,
    pace_diff_max: float = 2.0,
    years=None,
    require_higher_seed_net_negative: bool = True,  # NEW: toggle NET condition
) -> pd.DataFrame:
    """
    Return the filtered subset for the lower-seeded metric:
      - abs(Team_Pace - Opp_Pace) <= pace_diff_max
      - Row is lower-seeded (Team_Seed > Opp_Seed)
      - If require_higher_seed_net_negative=True:
          Higher-seeded NET_Diff < 0  ==> row NET_Diff > 0
    """
    df = _apply_years(df.copy(), years)
    df["Pace_Diff"] = (df["Team_Pace"] - df["Opp_Pace"]).abs()
    is_lower = df["Team_Seed"] > df["Opp_Seed"]

    mask = (df["Pace_Diff"] <= pace_diff_max) & is_lower
    if require_higher_seed_net_negative:
        mask = mask & (df["NET_Diff"] > 0)

    return df[mask]

def compute_filtered_df(
    df: pd.DataFrame,
    pace_diff_max: float = 2.0,
    require_higher_seed: bool = True,
    require_positive_net: bool = True,
    years=None,
) -> pd.DataFrame:
    """
    Return the filtered subset used for compute_win_pct() with matching flags.
    Use require_higher_seed=True and require_positive_net=True for your first metric;
    for the lower-seed metric, call this function with require_higher_seed=False and adjust logic as needed,
    or filter with your own mask.
    """
    df = _apply_years(df.copy(), years)
    df["Pace_Diff"] = (df["Team_Pace"] - df["Opp_Pace"]).abs()
    mask = df["Pace_Diff"] <= pace_diff_max

    if require_higher_seed:
        is_higher = df["Team_Seed"] < df["Opp_Seed"]
        mask &= is_higher
    if require_positive_net:
        mask &= df["NET_Diff"] > 0

    return df[mask]
