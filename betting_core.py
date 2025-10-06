# betting_core.py
import re
import math
import pandas as pd

# --- Basic team alias map (extend as you encounter more)
TEAM_ALIASES = {
    # Abbreviations / short names → canonical names matching your main dataset TeamName/OpponentName
    "ISU": "Iowa State",
    "Iowa St": "Iowa State",
    "Iowa St.": "Iowa State",
    "MSU": "Michigan State",
    "Marquette": "Marquette",
    "New Mexico": "New Mexico",
    "Bryant": "Bryant",
    "Lipscomb": "Lipscomb",
    "SMC": "Saint Mary's",
}

# --- Replace canonicalize_team with this version ---
def canonicalize_team(name: str) -> str:
    """
    Light canonicalization for a single team label:
    - trim, collapse spaces
    - normalize hyphens and MINUS (−, –, —) to '-'
    - remove rank/seed tokens like 'No. 5', '#12'
    - strip common suffixes (University/Univ/College)
    - apply alias mapping on final token if exact match
    """
    if pd.isna(name):
        return name
    s = str(name).strip()

    # Normalize various dashes/minus signs to simple hyphen
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")

    # Remove rank/seed/labels e.g., "No. 5", "#12", "Seed 7"
    s = re.sub(r"\b(no\.?|#)\s*\d+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bseed\s*\d+\b", "", s, flags=re.IGNORECASE)

    # Remove common suffix words
    s = re.sub(r"\b(university|univ|college)\b", "", s, flags=re.IGNORECASE)

    # Collapse whitespace / tidy punctuation
    s = re.sub(r"[\.']", "", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Alias exact (case-insensitive)
    for k, v in TEAM_ALIASES.items():
        if s.lower() == k.lower():
            return v
    return s

def american_to_implied_prob(odds: float) -> float:
    """Return implied probability (0-1) from American odds (+146, -178, etc.)."""
    if odds is None or pd.isna(odds):
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return -odds / (-odds + 100.0)

def american_payout(stake: float, odds: float) -> float:
    """
    Profit (not return) for a winning bet of `stake` dollars at American odds.
    e.g., +146 with $100 stake returns $146 profit, -178 returns $56.18 profit.
    """
    if odds > 0:
        return stake * (odds / 100.0)
    else:
        return stake * (100.0 / abs(odds))

# --- Replace parse_american_odds with this version ---
def parse_american_odds(cell: str):
    """
    Extract (team_name, odds:int, source:str|None).
    Handles Unicode MINUS and parentheses sources.
    Examples:
      'Marquette −178' -> ('Marquette', -178, None)
      'Lipscomb +760 (SB Nation)' -> ('Lipscomb', +760, 'SB Nation')
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return (None, None, None)

    s = str(cell).strip()

    # Extract trailing '(source)'
    source = None
    m_source = re.search(r"\(([^)]*)\)\s*$", s)
    if m_source:
        source = m_source.group(1).strip()
        s = s[:m_source.start()].strip()

    # Normalize unicode minus/en/em dash to '-'
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")

    # Split team + odds
    m = re.match(r"(.+?)\s*([+-]\d+)\s*$", s)
    if not m:
        return (None, None, source)

    team_raw = m.group(1).strip()
    odds_raw = int(m.group(2))
    team = canonicalize_team(team_raw)
    return (team, odds_raw, source)

def parse_score(cell: str):
    """Parse '82–55' or '82-55' (en dash/em dash/hyphen) -> (82, 55)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return (None, None)
    s = str(cell).strip().replace("–", "-").replace("—", "-")
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


def parse_point_spread(cell: str):
    """
    Return (favorite_team, spread_float).

    - Accepts 'Clemson -8.5', 'Clemson −8.5', 'Clemson 8.5' (rare exports),
      or malformed 'Clemson -  8.5' / 'Clemson -' + '8.5' style strings.
    - Strips any trailing '-', '+', '−', '–', '—' after the team name.
    - Final convention: favorite LAYS points -> spread is NEGATIVE.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return (None, None)

    s = str(cell).strip()
    # Normalize dash variants to a simple hyphen to make regex stable
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")

    # Typical formats we support:
    #   'Team -8.5'  'Team +8.5'  'Team 8.5'  'Team - 8.5'  'Team-8.5'
    m = re.match(r"(.+?)\s*[-+]?\s*([0-9]+(?:\.[0-9]+)?)\s*$", s)
    if not m:
        # Last-ditch: 'Team -' with number elsewhere (highly malformed)
        m2 = re.match(r"(.+?)\s*[-+]\s*$", s)
        if m2:
            team_raw = m2.group(1).strip()
            return (canonicalize_team(team_raw), None)
        return (None, None)

    team_raw = m.group(1).strip()
    spread = float(m.group(2))

    # If the team token ends with any sign, strip it
    while team_raw and team_raw[-1] in ("-", "+", "−", "–", "—"):
        team_raw = team_raw[:-1].strip()

    team = canonicalize_team(team_raw)

    # Enforce sign convention at parse time:
    # favorite should lay points, so store as NEGATIVE
    spread = -abs(spread)

    return (team, spread)


def normalize_team_token(name: str) -> str:
    """
    Aggressive normalization for fuzzy join:
    - remove seeds/tags like 'No. 5', '#12'
    - drop 'University/Univ', 'College'
    - normalize '&' -> 'and'
    - remove non-alphanumerics
    - lowercase
    """
    if pd.isna(name):
        return ""
    s = canonicalize_team(name)
    s = re.sub(r"\b(no\.?|#)\s*\d+\b", "", s, flags=re.IGNORECASE)  # No. 5 / #12
    s = re.sub(r"\bseed\s*\d+\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\b(university|univ|college)\b", "", s, flags=re.IGNORECASE)
    s = s.replace("&", "and")
    s = re.sub(r"[\.\-–—']", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^a-z0-9]", "", s)  # keep a-z0-9 only
    return s


def build_matchup_key(a: str, b: str) -> str:
    """Order-independent key for joining lines to games."""
    if a is None or b is None:
        return None
    ca, cb = canonicalize_team(a), canonicalize_team(b)
    return " :: ".join(sorted([ca, cb], key=lambda x: x.lower()))

def build_norm_matchup_key(a: str, b: str) -> str:
    if a is None or b is None:
        return None
    na, nb = normalize_team_token(a), normalize_team_token(b)
    return " :: ".join(sorted([na, nb]))

def load_betting_sheet(path: str, year_col: str = "Year") -> pd.DataFrame:
    """
    Expect columns (case-sensitive):
      - Matchup  (e.g., 'Iowa State vs. Lipscomb')
      - Point Spread
      - Moneyline (Favorite)
      - Moneyline (Underdog)
      - Year   (int)  <-- recommended to avoid cross-year collisions
    """
    raw = pd.read_excel(path, sheet_name=0)
    df = raw.copy()

    # Parse matchup teams
    m = df["Matchup"].astype(str).str.split(r"\s+vs\.?\s+", regex=True, expand=True)
    df["TeamA"] = m[0].map(canonicalize_team)
    df["TeamB"] = m[1].map(canonicalize_team)

    # Point spread
    fav_spread_team, fav_spread = zip(*df["Point Spread"].map(parse_point_spread))
    df["Spread_Fav_Team"] = list(fav_spread_team)
    df["Spread"] = list(fav_spread)

    # Moneylines (favorite / underdog)
    fav_team_ml, fav_odds, src_fav = zip(*df["Moneyline (Favorite)"].map(parse_american_odds))
    dog_team_ml, dog_odds, src_dog = zip(*df["Moneyline (Underdog)"].map(parse_american_odds))
    df["ML_Fav_Team"] = list(fav_team_ml)
    df["ML_Fav_Odds"] = list(fav_odds)
    df["ML_Dog_Team"] = list(dog_team_ml)
    df["ML_Dog_Odds"] = list(dog_odds)
    df["ML_Source"] = [sf or sd for sf, sd in zip(src_fav, src_dog)]

    # Keys (exact + normalized)
    df["matchup_key"] = [build_matchup_key(a, b) for a, b in zip(df["TeamA"], df["TeamB"])]
    df["norm_matchup_key"] = [build_norm_matchup_key(a, b) for a, b in zip(df["TeamA"], df["TeamB"])]

    # --- Reconcile ML team strings to TeamA/TeamB when possible ---
    df["_A_norm"] = df["TeamA"].map(normalize_team_token)
    df["_B_norm"] = df["TeamB"].map(normalize_team_token)

    def reconcile_ml_row(row):
        fav_team, fav_odds = row["ML_Fav_Team"], row["ML_Fav_Odds"]
        dog_team, dog_odds = row["ML_Dog_Team"], row["ML_Dog_Odds"]

        # If both odds exist and only one is negative, enforce favorite=negative, dog=positive
        if pd.notna(fav_odds) and pd.notna(dog_odds):
            if fav_odds > 0 and dog_odds < 0:
                fav_team, dog_team = dog_team, fav_team
                fav_odds, dog_odds = dog_odds, fav_odds

        a_norm, b_norm = row["_A_norm"], row["_B_norm"]

        def align(name):
            if pd.isna(name): return name
            n = normalize_team_token(name)
            if n == a_norm: return row["TeamA"]
            if n == b_norm: return row["TeamB"]
            return canonicalize_team(name)

        fav_team = align(fav_team)
        dog_team = align(dog_team)
        return pd.Series({
            "ML_Fav_Team": fav_team, "ML_Fav_Odds": fav_odds,
            "ML_Dog_Team": dog_team, "ML_Dog_Odds": dog_odds
        })

    reconciled = df.apply(reconcile_ml_row, axis=1)
    df["ML_Fav_Team"] = reconciled["ML_Fav_Team"]
    df["ML_Fav_Odds"] = reconciled["ML_Fav_Odds"]
    df["ML_Dog_Team"] = reconciled["ML_Dog_Team"]
    df["ML_Dog_Odds"] = reconciled["ML_Dog_Odds"]

    # Enforce spread sign convention: favorite lays points => negative spread value
    # If Spread_Fav_Team is known and spread is positive, flip sign.
    def normalize_spread_sign(row):
        sp = row["Spread"]
        if pd.isna(sp): return sp
        # If we have a favorite team, ensure negative number
        if pd.notna(row["Spread_Fav_Team"]):
            return -abs(float(sp))
        # Otherwise leave as-is (rare; can happen on malformed rows)
        return float(sp)

    df["Spread"] = df.apply(normalize_spread_sign, axis=1)

    # Implied probabilities
    df["ML_Fav_Implied"] = df["ML_Fav_Odds"].map(american_to_implied_prob)
    df["ML_Dog_Implied"] = df["ML_Dog_Odds"].map(american_to_implied_prob)

        # Scores (TeamA first, TeamB second)
    if "Score" in df.columns:
        a_pts, b_pts = zip(*df["Score"].map(parse_score))
        df["TeamA_Points"] = list(a_pts)
        df["TeamB_Points"] = list(b_pts)
    else:
        df["TeamA_Points"] = pd.NA
        df["TeamB_Points"] = pd.NA

    # Clean temp cols
    df = df.drop(columns=["_A_norm", "_B_norm"])

    # Year: coerce to numeric Int64
    if year_col in df.columns:
        df["Year"] = df[year_col]
    else:
        df["Year"] = pd.NA
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    keep = [
        "Year", "matchup_key", "norm_matchup_key",
        "TeamA", "TeamB",
        "Spread_Fav_Team", "Spread",
        "ML_Fav_Team", "ML_Fav_Odds", "ML_Fav_Implied",
        "ML_Dog_Team", "ML_Dog_Odds", "ML_Dog_Implied",
        "TeamA_Points", "TeamB_Points",
        "ML_Source",
    ]
    return df[keep]


def attach_betting_to_games(games_df: pd.DataFrame, lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    games_df: your per-row team data (two rows per game). Must contain columns TeamName, OpponentName, Year.
    We attach betting lines on an order-independent join (matchup_key + Year), then re-orient to the row's perspective.
    """
    g = games_df.copy()
    # ✅ Ensure Year is numeric (nullable Int64) on the games side
    g["Year"] = pd.to_numeric(g["Year"], errors="coerce").astype("Int64")
    g["matchup_key"] = [build_matchup_key(a, b) for a, b in zip(g["TeamName"], g["OpponentName"])]

    merged = g.merge(
        lines_df,
        on=["Year", "matchup_key"],
        how="left",
        suffixes=("", "_lines"),
    )

        # Fallback pass: normalized key for rows still missing
    # Fallback pass: normalized key for rows still missing
    missing_mask = merged["ML_Fav_Odds"].isna() & merged["ML_Dog_Odds"].isna() & merged["Spread"].isna()
    if missing_mask.any():
        g_missing = merged.loc[missing_mask].copy()

        # Ensure we have norm_matchup_key on the left (recompute if missing)
        if "norm_matchup_key" not in g_missing.columns or g_missing["norm_matchup_key"].isna().any():
            g_missing["norm_matchup_key"] = [
                build_norm_matchup_key(a, b)
                for a, b in zip(g_missing["TeamName"], g_missing["OpponentName"])
            ]

        # Only the columns we need from the right
        right_cols = [
            "Year", "norm_matchup_key",
            "Spread_Fav_Team", "Spread",
            "ML_Fav_Team", "ML_Fav_Odds", "ML_Fav_Implied",
            "ML_Dog_Team", "ML_Dog_Odds", "ML_Dog_Implied",
            "TeamA_Points", "TeamB_Points",
            "ML_Source",
        ]
        rr = lines_df[right_cols].copy()

        alt = g_missing.merge(
            rr, how="left",
            on=["Year", "norm_matchup_key"],
            suffixes=("", "_alt")
        )

        # Fill only where empty
        fill_cols = [
            "Spread_Fav_Team", "Spread",
            "ML_Fav_Team", "ML_Fav_Odds", "ML_Fav_Implied",
            "ML_Dog_Team", "ML_Dog_Odds", "ML_Dog_Implied",
            "TeamA_Points", "TeamB_Points",
            "ML_Source",
        ]
        for col in fill_cols:
            merged.loc[missing_mask, col] = merged.loc[missing_mask, col].combine_first(alt[col])



    # Determine, from the row's perspective, the row's moneyline odds
    def row_ml_odds(row):
        team = canonicalize_team(row["TeamName"])
        if pd.notna(row.get("ML_Fav_Team")) and team == row["ML_Fav_Team"]:
            return row["ML_Fav_Odds"]
        if pd.notna(row.get("ML_Dog_Team")) and team == row["ML_Dog_Team"]:
            return row["ML_Dog_Odds"]
        return pd.NA

    def row_ml_implied(row):
        team = canonicalize_team(row["TeamName"])
        if pd.notna(row.get("ML_Fav_Team")) and team == row["ML_Fav_Team"]:
            return row["ML_Fav_Implied"]
        if pd.notna(row.get("ML_Dog_Team")) and team == row["ML_Dog_Team"]:
            return row["ML_Dog_Implied"]
        return pd.NA

    merged["Row_ML_Odds"] = merged.apply(row_ml_odds, axis=1)
    merged["Row_ML_Implied"] = merged.apply(row_ml_implied, axis=1)

    # Simple profit calc for $100 stake if you bet this row's team moneyline
    def row_profit_if_bet_100(row):
        odds = row["Row_ML_Odds"]
        won = row.get("Team_Won", pd.NA)
        if pd.isna(odds) or pd.isna(won):
            return pd.NA
        if won == 1:
            return american_payout(100.0, odds)
        else:
            return -100.0

    merged["Row_ML_Profit_$100"] = merged.apply(row_profit_if_bet_100, axis=1)

    # ----- ATS (Against the Spread) -----
    # Determine row's team score and opponent score from TeamA/TeamB points
    def row_points(row):
        t = canonicalize_team(row["TeamName"])
        if pd.notna(row.get("TeamA")) and pd.notna(row.get("TeamA_Points")):
            if t == row["TeamA"]:
                return row["TeamA_Points"], row["TeamB_Points"]
            if t == row["TeamB"]:
                return row["TeamB_Points"], row["TeamA_Points"]
        return (pd.NA, pd.NA)

    pts = merged.apply(row_points, axis=1, result_type="expand")
    merged["Row_Team_Points"] = pts[0]
    merged["Row_Opp_Points"] = pts[1]

    # Ensure spread stored is negative for favorite (done in loader), then compute row-specific spread:
    # If the row's team IS the favorite, row_spread = Spread (negative).
    # Else row_spread = -Spread (positive points received).
    def row_spread_value(row):
        sp = row.get("Spread", pd.NA)
        fav = row.get("Spread_Fav_Team", pd.NA)
        if pd.isna(sp) or pd.isna(fav):
            return pd.NA
        if canonicalize_team(row["TeamName"]) == fav:
            return float(sp)  # negative
        else:
            return -float(sp)  # positive (dog gets points)

    merged["Row_Spread"] = merged.apply(row_spread_value, axis=1)

    def ats_result(row):
        rp, op, rs = row.get("Row_Team_Points"), row.get("Row_Opp_Points"), row.get("Row_Spread")
        if any(pd.isna(x) for x in [rp, op, rs]):
            return pd.NA
        adj_margin = (rp - op) + rs
        if adj_margin > 0:  return "ATS Win"
        if adj_margin == 0: return "Push"
        return "ATS Loss"

    merged["Row_ATS_Result"] = merged.apply(ats_result, axis=1)


    return merged
