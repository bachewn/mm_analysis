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

def parse_point_spread(cell: str):
    """
    Extract (favorite_team, spread_float). Example:
      'ISU −14.5' → ('Iowa State', -14.5)  # negative means favorite giving points
      'Marquette -3.5' → ('Marquette', -3.5)
    """
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return (None, None)
    s = str(cell).strip().replace("–", "-").replace("—", "-")
    m = re.match(r"(.+?)\s*([+-]?\d+(\.\d+)?)\s*$", s)
    if not m:
        return (None, None)
    team_raw = m.group(1).strip()
    spread = float(m.group(2))
    # Convention: favorite has negative spread; if positive provided, make it negative (favorite)
    # Some books use "Team +3.5 (underdog)"—we keep the sign as-is (user can interpret),
    # but most first-round lines will show favorite with negative.
    team = canonicalize_team(team_raw)
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

    # Implied probabilities
    df["ML_Fav_Implied"] = df["ML_Fav_Odds"].map(american_to_implied_prob)
    df["ML_Dog_Implied"] = df["ML_Dog_Odds"].map(american_to_implied_prob)

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

    return merged
