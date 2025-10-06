# streamlit_app.py
import os
import math
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from urllib.parse import urlencode
from betting_core import load_betting_sheet, attach_betting_to_games


from mm_core import (
    load_dataset,
    compute_win_pct,
    compute_lower_seed_win_pct,
    compute_filtered_df,
    compute_lower_seed_filtered_df,
)

# ---------- Page config ----------
st.set_page_config(page_title="MM First-Round — Explorer", layout="wide")

# ---------- Utilities ----------
def set_query_params(**kwargs):
    st.query_params.clear()
    for k, v in kwargs.items():
        if v is None:
            continue
        st.query_params[k] = str(v)

def get_qp(name, default=None, cast=str):
    val = st.query_params.get(name, None)
    if isinstance(val, list):
        val = val[0]
    if val is None:
        return default
    try:
        return cast(val)
    except Exception:
        return default

def get_bool01(name, default=False):
    val = get_qp(name, None, str)
    if val is None:
        return default
    val = str(val).lower()
    if val in ("1","true","yes","on"):
        return True
    if val in ("0","false","no","off"):
        return False
    return default

def badge(text, kind="neutral"):
    colors = {"neutral":"#e5e7eb","success":"#22c55e","danger":"#ef4444","info":"#3b82f6","warn":"#f59e0b"}
    return f"<span style='background:{colors.get(kind,'#e5e7eb')}; color:#111827; padding:2px 8px; border-radius:999px; font-size:12px;'>{text}</span>"

def pretty_card(title, body_md):
    st.markdown(
        f"""
        <div style="border:1px solid #eee;border-radius:16px;padding:16px;margin:6px 0;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,0.05)">
          <div style="font-weight:600;font-size:16px;margin-bottom:6px">{title}</div>
          <div style="font-size:14px;color:#374151">{body_md}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def detect_pair(df, team_col, opp_col):
    return team_col in df.columns and opp_col in df.columns

def diff_series(df, team_col, opp_col, absolute=False):
    s = df[team_col] - df[opp_col]
    return s.abs() if absolute else s

def build_advanced_filters_ui(df):
    """Return dict of active filters based on user input (off by default)."""
    active = {}
    with st.sidebar.expander("Advanced filters", expanded=False):
        st.caption("Optional — only applied when enabled. You can use |Δ| or signed Δ for each metric.")

        # Turnovers
        if detect_pair(df, "Team_Turnovers", "Opp_Turnovers"):
            enable_tov = st.checkbox("Filter by Turnover differential", value=False, key="tov_en")
            if enable_tov:
                tov_abs = st.radio(
                    "Turnover type",
                    ["Absolute |team − opp|", "Signed (team − opp)"],
                    index=0,
                    key="tov_abs"
                ) == "Absolute |team − opp|"
                series = diff_series(df, "Team_Turnovers", "Opp_Turnovers", absolute=tov_abs)
                lo, hi = float(series.min()), float(series.max())
                step = max((hi - lo) / 100, 1.0) if hi > lo else 1.0
                sel = st.slider(
                    "Turnover differential range",
                    min_value=lo, max_value=hi, value=(lo, hi), step=step, key="tov_rng"
                )
                active["tov"] = {"abs": tov_abs, "min": float(sel[0]), "max": float(sel[1])}

        # Rebounds
        if detect_pair(df, "Team_Rebounds", "Opp_Rebounds"):
            enable_reb = st.checkbox("Filter by Rebound differential", value=False, key="reb_en")
            if enable_reb:
                reb_abs = st.radio(
                    "Rebound type",
                    ["Absolute |team − opp|", "Signed (team − opp)"],
                    index=0,
                    key="reb_abs"
                ) == "Absolute |team − opp|"
                series = diff_series(df, "Team_Rebounds", "Opp_Rebounds", absolute=reb_abs)
                lo, hi = float(series.min()), float(series.max())
                step = max((hi - lo) / 100, 1.0) if hi > lo else 1.0
                sel = st.slider(
                    "Rebound differential range",
                    min_value=lo, max_value=hi, value=(lo, hi), step=step, key="reb_rng"
                )
                active["reb"] = {"abs": reb_abs, "min": float(sel[0]), "max": float(sel[1])}

        # eFG% (auto-detect common column names)
        efg_pairs = [("Team_EFG", "Opp_EFG"), ("Team_EFG_Pct", "Opp_EFG_Pct"), ("Team_EFG%", "Opp_EFG%")]
        tcol = ocol = None
        for t, o in efg_pairs:
            if detect_pair(df, t, o):
                tcol, ocol = t, o
                break
        if tcol and ocol:
            enable_efg = st.checkbox("Filter by Effective FG% differential", value=False, key="efg_en")
            if enable_efg:
                efg_abs = st.radio(
                    "eFG% type",
                    ["Absolute |team − opp|", "Signed (team − opp)"],
                    index=0,
                    key="efg_abs"
                ) == "Absolute |team − opp|"
                series = diff_series(df, tcol, ocol, absolute=efg_abs)
                lo, hi = float(series.min()), float(series.max())
                step = max((hi - lo) / 100, 0.5) if hi > lo else 0.5
                sel = st.slider(
                    "eFG% differential range",
                    min_value=lo, max_value=hi, value=(lo, hi), step=step, key="efg_rng"
                )
                active["efg"] = {"abs": efg_abs, "min": float(sel[0]), "max": float(sel[1]), "cols": (tcol, ocol)}

    return active

def apply_advanced_filters(df, filters):
    if not filters:
        return df
    m = pd.Series(True, index=df.index)
    if "tov" in filters and detect_pair(df, "Team_Turnovers", "Opp_Turnovers"):
        p = filters["tov"]
        s = diff_series(df, "Team_Turnovers", "Opp_Turnovers", absolute=p["abs"])
        m &= (s >= p["min"]) & (s <= p["max"])
    if "reb" in filters and detect_pair(df, "Team_Rebounds", "Opp_Rebounds"):
        p = filters["reb"]
        s = diff_series(df, "Team_Rebounds", "Opp_Rebounds", absolute=p["abs"])
        m &= (s >= p["min"]) & (s <= p["max"])
    if "efg" in filters:
        p = filters["efg"]
        tcol, ocol = p.get("cols", ("Team_EFG", "Opp_EFG"))
        if detect_pair(df, tcol, ocol):
            s = diff_series(df, tcol, ocol, absolute=p["abs"])
            m &= (s >= p["min"]) & (s <= p["max"])
    return df[m]

def encode_filters_qp(filters: dict) -> dict:
    qp = {}
    if "tov" in filters:
        qp.update({"tov":1, "tov_abs": int(filters["tov"]["abs"]), "tov_min": filters["tov"]["min"], "tov_max": filters["tov"]["max"]})
    if "reb" in filters:
        qp.update({"reb":1, "reb_abs": int(filters["reb"]["abs"]), "reb_min": filters["reb"]["min"], "reb_max": filters["reb"]["max"]})
    if "efg" in filters:
        qp.update({"efg":1, "efg_abs": int(filters["efg"]["abs"]), "efg_min": filters["efg"]["min"], "efg_max": filters["efg"]["max"]})
    return qp

def decode_filters_qp(df) -> dict:
    filters = {}
    if get_bool01("tov", False) and detect_pair(df, "Team_Turnovers", "Opp_Turnovers"):
        filters["tov"] = {
            "abs": bool(get_qp("tov_abs", 1, int)),
            "min": float(get_qp("tov_min", -math.inf, float)),
            "max": float(get_qp("tov_max", math.inf, float)),
        }
    if get_bool01("reb", False) and detect_pair(df, "Team_Rebounds", "Opp_Rebounds"):
        filters["reb"] = {
            "abs": bool(get_qp("reb_abs", 1, int)),
            "min": float(get_qp("reb_min", -math.inf, float)),
            "max": float(get_qp("reb_max", math.inf, float)),
        }
    if get_bool01("efg", False):
        efg_pairs = [("Team_EFG", "Opp_EFG"), ("Team_EFG_Pct", "Opp_EFG_Pct"), ("Team_EFG%", "Opp_EFG%")]
        for tcol, ocol in efg_pairs:
            if detect_pair(df, tcol, ocol):
                filters["efg"] = {
                    "abs": bool(get_qp("efg_abs", 1, int)),
                    "min": float(get_qp("efg_min", -math.inf, float)),
                    "max": float(get_qp("efg_max", math.inf, float)),
                    "cols": (tcol, ocol),
                }
                break
    return filters

# ---------- Load data ----------
DATA_PATH = os.getenv("MM_DATA_PATH", "/data/df_first_round_all_years.xlsx")
df = load_dataset(DATA_PATH)
years_available = sorted(df["Year"].unique())
LINES_PATH = os.getenv("MM_LINES_PATH")  # optional
lines_df = None
if LINES_PATH and os.path.exists(LINES_PATH):
    try:
        lines_df = load_betting_sheet(LINES_PATH)
    except Exception as e:
        st.sidebar.warning(f"Could not load betting lines: {e}")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
if lines_df is not None:
    lines_df["Year"] = pd.to_numeric(lines_df["Year"], errors="coerce").astype("Int64")

# ---------- Sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    sel_years = st.multiselect("Years", years_available, default=years_available)
    pace_diff = st.slider("Max Pace Differential", min_value=0.0, max_value=12.0, value=3.0, step=0.5)
    require_net = st.checkbox(
        "Require Strength of Schedule (SoS) condition",
        value=True,
        help=(
            "When ON: Higher-seed metric requires SoS > 0 on the higher-seed row. "
            "Lower-seed metric requires the higher seed's SoS < 0 (i.e., SoS > 0 on the lower-seed row). "
            "When OFF: SoS condition is not applied."
        ),
    )
    # NEW: optional advanced filters (off by default)
    adv_filters = build_advanced_filters_ui(df)

# ---------- Routing ----------
view = get_qp("view", "list", str)
detail_mode = get_qp("mode", None, str)   # 'higher' or 'lower'
detail_year = get_qp("year", None, int)

# read snapshot filters from query params (for details page)
detail_pace = get_qp("pace", pace_diff, float)
detail_require_net = get_bool01("net", require_net)
qp_adv = decode_filters_qp(df)

# ---------- Apply advanced filters to a working copy ----------
df_adv = apply_advanced_filters(df, adv_filters if view == "list" else qp_adv)

df_with_lines = attach_betting_to_games(df_adv, lines_df) if lines_df is not None else df_adv


# ---------- Detail Page ----------
if view == "detail" and detail_mode in {"higher", "lower"} and detail_year is not None:
    st.title("First-Round Details")
    chips = [f"Year: {detail_year}", f"Pace ≤ {detail_pace:g}", "NET required" if detail_require_net else "NET not required"]
    if qp_adv:
        if "tov" in qp_adv: chips.append(f"TOV {'|Δ|' if qp_adv['tov']['abs'] else 'Δ'} in [{qp_adv['tov']['min']:.2f}, {qp_adv['tov']['max']:.2f}]")
        if "reb" in qp_adv: chips.append(f"REB {'|Δ|' if qp_adv['reb']['abs'] else 'Δ'} in [{qp_adv['reb']['min']:.2f}, {qp_adv['reb']['max']:.2f}]")
        if "efg" in qp_adv: chips.append(f"eFG {'|Δ|' if qp_adv['efg']['abs'] else 'Δ'} in [{qp_adv['efg']['min']:.2f}, {qp_adv['efg']['max']:.2f}]")
    st.markdown("**Filters (snapshot):** " + " • ".join(chips))
    st.markdown("---")

    df_y = df_with_lines[df_with_lines["Year"] == detail_year].copy()

    if detail_mode == "higher":
        filt = compute_filtered_df(
            df_y,
            pace_diff_max=detail_pace,
            require_higher_seed=True,
            require_positive_net=detail_require_net,
            years=[detail_year],
        )
        pick_side = "Higher-seed pick"
    else:
        filt = compute_lower_seed_filtered_df(
            df_y,
            pace_diff_max=detail_pace,
            years=[detail_year],
            require_higher_seed_net_negative=detail_require_net,
        )
        pick_side = "Lower-seed pick"

    n = len(filt); wins = int(filt["Team_Won"].sum()); losses = n - wins; w_pct = (wins / n * 100.0) if n else 0.0
    pretty_card(f"{pick_side}: {badge(f'{w_pct:.1f}% W','success')}", f"{badge(f'n={n}','neutral')} • {badge(f'wins={wins}','success')} • {badge(f'losses={losses}','danger')}")

    def game_label(row):
        return f"{row['TeamName']} (#{int(row['Team_Seed'])}) vs {row['OpponentName']} (#{int(row['Opp_Seed'])})"

    if n == 0:
        st.info("No games matched your filters for this year.")
    else:
        disp = filt.copy()
        if "Pace_Diff" not in disp.columns and {'Team_Pace','Opp_Pace'}.issubset(disp.columns):
            disp["Pace_Diff"] = (disp["Team_Pace"] - disp["Opp_Pace"]).abs()
        disp["Game"] = disp.apply(game_label, axis=1)
        disp["Pick"] = disp["TeamName"]
        disp["Result"] = disp["Team_Won"].map(lambda x: "Win ✅" if x == 1 else "Loss ❌")
        disp_cols = [
            "Year", "Game", "Pick", "Result",
            "NET_Diff", "Pace_Diff",
            # Betting:
            "Spread_Fav_Team", "Spread",
            "ML_Fav_Team", "ML_Fav_Odds",
            "ML_Dog_Team", "ML_Dog_Odds",
            "Row_ML_Odds", "Row_ML_Implied", "Row_ML_Profit_$100",
            # Scores + ATS
            "Row_Team_Points", "Row_Opp_Points", "Row_Spread", "Row_ATS_Result",
        ]

        present = [c for c in disp_cols if c in disp.columns]
        st.dataframe(disp[present].sort_values(by=["Result","Game"], ascending=[True,True]).reset_index(drop=True), width="stretch")
        st.download_button("Download detailed results (CSV)", data=disp[present].to_csv(index=False), file_name=f"{detail_mode}_detail_{detail_year}.csv", mime="text/csv")

    st.markdown("—")
    if st.button("← Back to overview"):
        set_query_params(view="list")
        st.rerun()
    st.stop()

# ---------- Overview Page ----------
st.title("March Madness First-Round — Explorer")

# Compute metrics on advanced-filtered dataframe
higher_res = compute_win_pct(df_adv, pace_diff_max=pace_diff, by_year=True, years=sel_years, return_counts=True, require_positive_net=require_net)
lower_res  = compute_lower_seed_win_pct(df_adv, pace_diff_max=pace_diff, by_year=True, years=sel_years, return_counts=True, require_higher_seed_net_negative=require_net)

# Top metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Higher-seed win % (Favored)", f"{higher_res['overall']['pct']:.1f}%", f"n={higher_res['overall']['n']}")
with col2:
    st.metric("Lower-seed win % (Underdog)", f"{lower_res['overall']['pct']:.1f}%", f"n={lower_res['overall']['n']}")
with col3:
    st.write("Pace ≤", pace_diff)
    st.write("Years:", ", ".join(map(str, sel_years)) if sel_years else "All")
st.markdown("---")

by_year_1 = pd.DataFrame(higher_res["by_year"])
by_year_2 = pd.DataFrame(lower_res["by_year"])

# Clickable year tables using data_editor LinkColumn (renders pretty links)
def to_link_table(df_years: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df_years.copy()
    def make_url(y):
        params = {
            "view": "detail",
            "mode": mode,
            "year": int(y),
            "net": int(bool(require_net)),    # snapshot of current toggles
            "pace": pace_diff,
        }
        params.update(encode_filters_qp(adv_filters))  # include advanced filter snapshot
        return f"/?{urlencode(params)}"
    out["Open"] = out["Year"].map(make_url)
    cols = ["Year", "n", "wins", "pct", "Open"] if set(["n","wins","pct"]).issubset(out.columns) else list(out.columns)+["Open"]
    return out[cols]

tbl_high = to_link_table(by_year_1, "higher")
tbl_low  = to_link_table(by_year_2, "lower")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Higher-Seeded Teams — Year Breakdown")
    st.data_editor(
        tbl_high, width="stretch", disabled=True, hide_index=True,
        column_config={
            "pct": st.column_config.NumberColumn("pct", format="%.1f"),
            "Open": st.column_config.LinkColumn("Open", display_text="Open")
        },
    )
with c2:
    st.subheader("Lower-Seeded Teams — Year Breakdown")
    st.data_editor(
        tbl_low, width="stretch", disabled=True, hide_index=True,
        column_config={
            "pct": st.column_config.NumberColumn("pct", format="%.1f"),
            "Open": st.column_config.LinkColumn("Open", display_text="Open")
        },
    )

# Combined chart (one plot)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(by_year_1["Year"], by_year_1["pct"], marker="o", label="Higher-seed %")
ax.plot(by_year_2["Year"], by_year_2["pct"], marker="o", label="Lower-seed %")
ax.set_title("Win % by Year (Filtered)")
ax.set_xlabel("Year"); ax.set_ylabel("Win %"); ax.set_ylim(0, 100); ax.legend()
st.pyplot(fig)

# Seed vs Seed matrix from advanced-filtered df (higher-seed perspective)
# Seed vs Seed matrix from advanced-filtered df (higher-seed perspective)
# Use betting-enriched df if available so we can compute ATS
tmp = (df_with_lines.copy() if 'df_with_lines' in locals() and isinstance(df_with_lines, pd.DataFrame)
       else df_adv.copy())

tmp["Pace_Diff"] = (tmp["Team_Pace"] - tmp["Opp_Pace"]).abs()
base_mask = (tmp["Pace_Diff"] <= pace_diff) & (tmp["Team_Seed"] < tmp["Opp_Seed"])  # higher-seed rows only
mask = (base_mask & (tmp["NET_Diff"] > 0)) if require_net else base_mask
f_high = tmp[mask]
if sel_years:
    f_high = f_high[f_high["Year"].isin(sel_years)]

# --- Win % matrix (straight-up, higher-seed perspective) ---
seed_mat = (
    f_high.groupby(["Team_Seed", "Opp_Seed"])
    .agg(n=("Team_Won", "size"), wins=("Team_Won", "sum"))
    .assign(pct=lambda d: d["wins"] / d["n"] * 100.0)
    .reset_index()
)
pivot_pct = seed_mat.pivot(index="Team_Seed", columns="Opp_Seed", values="pct").sort_index().sort_index(axis=1)
pivot_n   = seed_mat.pivot(index="Team_Seed", columns="Opp_Seed", values="n").sort_index().sort_index(axis=1)

st.subheader("Seed vs Seed — Higher-Seed Perspective (filtered)")
st.caption("Rows: higher seed; Cols: lower seed. pct = higher seed win %. ATS: Against-the-spread (point-spread)")

t1, t2, t3 = st.tabs(["Win % matrix", "Sample size (n) matrix", "ATS % matrix"])

with t1:
    st.dataframe(pivot_pct.style.format("{:.1f}"), use_container_width=True)

with t2:
    st.dataframe(pivot_n.fillna(0).astype(int), use_container_width=True)

with t3:
    # ATS matrix: require Row_ATS_Result (from betting lines attachment)
    if "Row_ATS_Result" not in f_high.columns:
        st.info("ATS data isn’t available yet (no betting lines attached).")
    else:
        ats = f_high.copy()
        # Count only wins/losses; exclude pushes from denominator
        ats["ats_win"]   = (ats["Row_ATS_Result"] == "ATS Win").astype(int)
        ats["ats_valid"] = ats["Row_ATS_Result"].isin(["ATS Win", "ATS Loss"]).astype(int)

        ats_mat = (
            ats.groupby(["Team_Seed", "Opp_Seed"])
            .agg(ats_wins=("ats_win", "sum"), ats_n=("ats_valid", "sum"))
            .assign(ats_pct=lambda d: (d["ats_wins"] / d["ats_n"] * 100.0).where(d["ats_n"] > 0))
            .reset_index()
        )

        pivot_ats_pct = (
            ats_mat.pivot(index="Team_Seed", columns="Opp_Seed", values="ats_pct")
            .sort_index().sort_index(axis=1)
        )
        pivot_ats_n = (
            ats_mat.pivot(index="Team_Seed", columns="Opp_Seed", values="ats_n")
            .sort_index().sort_index(axis=1)
        )

        c_ats1, c_ats2 = st.columns(2)
        with c_ats1:
            st.markdown("**ATS win % (higher-seed perspective)**")
            st.dataframe(pivot_ats_pct.style.format(lambda v: "" if pd.isna(v) else f"{v:.1f}"), use_container_width=True)
        with c_ats2:
            st.markdown("**ATS sample size (valid bets only)**")
            st.dataframe(pivot_ats_n.fillna(0).astype(int), use_container_width=True)

        st.caption("ATS % excludes pushes from the denominator. Cells with no valid ATS bets are blank.")
