import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://kenpom.com/index.php?y=2019"
OUTFILE = "kenpom_2019.xlsx"

# Optional: set a friendly header
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36"
}
resp = requests.get("https://kenpom.com/index.php?y=2019", headers=headers)
resp.raise_for_status()

soup = BeautifulSoup(resp.text, "lxml")

# KenPom renders a single big ratings table. We’ll parse the visible rows robustly.
rows = []

# Grab all text rows that start with a rank number followed by a team link
for tr in soup.select("tr"):
    tds = tr.find_all("td")
    if not tds:
        continue

    # Expect the row to start with rank and team, etc.
    # The rating table generally has at least ~10+ columns.
    if not tds[0].get_text(strip=True).isdigit():
        continue

    try:
        # Basic fields
        rk = int(tds[0].get_text(strip=True))
        team = tds[1].get_text(" ", strip=True)
        conf = tds[2].get_text(strip=True)

        wl = tds[3].get_text(strip=True)  # e.g., "35-3"
        m = re.match(r"(\d+)-(\d+)", wl)
        w, l = (int(m.group(1)), int(m.group(2))) if m else (None, None)

        # Numeric columns (KenPom usual order on ratings page):
        # AdjEM, AdjO, AdjD, AdjT, Luck, SOS AdjEM, SOS OppO, SOS OppD, NCSOS AdjEM
        def to_float(x):
            x = x.replace("−", "-")  # normalize minus if needed
            return float(x)

        # The page shows ranks beside some stats; cells often look like "123.4 (2)".
        # We’ll strip parenthetical ranks when present.
        def split_val_rank(text):
            # examples: "123.4", "123.4 (2)", "+.050", "-3.24", "59.4 (353)"
            t = text.replace("\xa0", " ").strip()
            # Handle leading "+." like "+.050" => "+0.050"
            t = re.sub(r'^\+\.', '+0.', t)
            t = re.sub(r'^-\.', '-0.', t)
            val = re.sub(r"\s*\(\d+\)\s*$", "", t)
            return to_float(val)

        # The KenPom ratings table layout at this URL (2019) is:
        # rk, team, conf, W-L,
        # AdjEM, AdjO, AdjD, AdjT, Luck,
        # SOS AdjEM, SOS OppO, SOS OppD, NCSOS AdjEM
        # (some cells include parenthetical ranks we strip away)
        AdjEM      = split_val_rank(tds[4].get_text())
        AdjO       = split_val_rank(tds[5].get_text())
        AdjD       = split_val_rank(tds[7].get_text()) if len(tds) > 7 else None
        AdjT       = split_val_rank(tds[9].get_text()) if len(tds) > 9 else None
        Luck       = split_val_rank(tds[11].get_text()) if len(tds) > 11 else None
        SOS_AdjEM  = split_val_rank(tds[12].get_text()) if len(tds) > 12 else None
        SOS_OppO   = split_val_rank(tds[13].get_text()) if len(tds) > 13 else None
        SOS_OppD   = split_val_rank(tds[15].get_text()) if len(tds) > 15 else None
        NCSOS_AdjEM= split_val_rank(tds[17].get_text()) if len(tds) > 17 else None

        rows.append({
            "Rk": rk,
            "Team": team,
            "Conf": conf,
            "W": w, "L": l,
            "AdjEM": AdjEM,
            "AdjO": AdjO,
            "AdjD": AdjD,
            "AdjT": AdjT,
            "Luck": Luck,
            "SOS AdjEM": SOS_AdjEM,
            "SOS OppO": SOS_OppO,
            "SOS OppD": SOS_OppD,
            "NCSOS AdjEM": NCSOS_AdjEM,
        })
    except Exception:
        # Skip any odd header/duplicate subheader rows
        continue

# Build DataFrame, sort by rank, and write to Excel
df = pd.DataFrame(rows).sort_values("Rk").reset_index(drop=True)

# Light sanity checks
assert df["Rk"].is_monotonic_increasing, "Ranks not monotonic — parsing offset likely changed."
assert df.shape[0] >= 300, f"Parsed fewer rows than expected: {df.shape[0]}"

df.to_excel(OUTFILE, index=False)
print(f"Saved {df.shape[0]} rows to {OUTFILE}")
