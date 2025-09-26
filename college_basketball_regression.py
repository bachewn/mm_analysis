import pandas as pd
import plotly.express as px
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
import numpy as np
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score


###############################################################################
#                           1. CONFIG & HELPERS
###############################################################################

#parameter search space
param_space = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'solver': Categorical(['liblinear', 'lbfgs']),  # or others like 'sag', 'saga'
}

# Adjust this dictionary or generate file paths dynamically:
DATA_PATHS = {
    "pace":       r"C:\Users\888\Desktop\2024 NCAA Stats\pace_{year}.xlsx",
    "season_rec": r"C:\Users\888\Desktop\2024 NCAA Stats\season_record_{year}.xlsx",
    "bracket":    r"C:\Users\888\Desktop\2024 NCAA Stats\bracket_results_{year}.xlsx",
    "net_rating":  r"C:\Users\888\Desktop\2024 NCAA Stats\ken_pom_net_rating_{year}.xlsx",
    "rebounds":  r"C:\Users\888\Desktop\2024 NCAA Stats\rebounds_{year}.xlsx",
    "offensive":  r"C:\Users\888\Desktop\2024 NCAA Stats\offensive_rating_{year}.xlsx",
    "defensive":  r"C:\Users\888\Desktop\2024 NCAA Stats\defensive_rating_{year}.xlsx",
    "adj_rating":  r"C:\Users\888\Desktop\2024 NCAA Stats\adjusted_rating_{year}.xlsx",
    "efg":  r"C:\Users\888\Desktop\2024 NCAA Stats\effective_field_goal_{year}.xlsx",
    "fta":  r"C:\Users\888\Desktop\2024 NCAA Stats\free_throw_attempt_{year}.xlsx",
    "turnovers":  r"C:\Users\888\Desktop\2024 NCAA Stats\turnovers_{year}.xlsx",
}

def get_filepath(file_type: str, year: int) -> str:
    """
    Return the file path for a given type ("pace", "season_rec", or "bracket")
    and year.
    """
    pattern = DATA_PATHS.get(file_type)
    return pattern.format(year=year)

###############################################################################
#                           2. DATA LOADING FUNCTIONS
###############################################################################

def load_and_prepare_season_data(year: int) -> pd.DataFrame:
    """
    Loads pace and win-loss data for the specified year, merges them,
    and calculates the Win-Loss Ratio.
    
    Returns a DataFrame with columns:
      - Name
      - wins
      - losses
      - Pace
      - Win-Loss Ratio
      ...
    """
    pace_df = pd.read_excel(get_filepath("pace", year))
    season_df = pd.read_excel(get_filepath("season_rec", year))

    # Convert columns to numeric
    season_df["wins"]   = pd.to_numeric(season_df["Wins"], errors="coerce")
    season_df["losses"] = pd.to_numeric(season_df["Losses"], errors="coerce")

    # Merge pace with season record
    combined = pd.merge(
        season_df, pace_df,
        on="Name", 
        how="inner"
    )

    # Calculate Win-Loss ratio
    combined["Win-Loss Ratio"] = combined["wins"] / (combined["wins"] + combined["losses"])
    
    return combined

def load_factors_for_year(year: int, factor_keys: list[str]) -> pd.DataFrame:
    """
    Loads multiple factor files (each must have 'Name' plus a factor column),
    merges them into one DataFrame keyed on 'Name'.
    
    Example of factor_keys: ['pace', 'net_rating', 'oreb_pct']
    This will produce a DataFrame with columns:
        Name, Pace, NET_Rating, OReb_Pct, ...
    """
    # Start with None; we'll do a chain-merge
    merged_df = None

    for fk in factor_keys:
        path = DATA_PATHS[fk].format(year=year)
        df_factor = pd.read_excel(path)
        # Clean up if needed, e.g. strip spaces from 'Name'
        df_factor['Name'] = df_factor['Name'].str.strip()
        
        # If this factor has a special column name, rename it to something standard
        # We can store these in a dictionary or rely on the file's column name
        if fk == 'pace':
            df_factor.rename(columns={'Pace': 'pace'}, inplace=True)
        elif fk == 'net_rating':
            df_factor.rename(columns={'Net_Rating': 'net_rating'}, inplace=True)
        elif fk == 'rebounds':
            df_factor.rename(columns={'Rebounds': 'rebounds'}, inplace=True)
        elif fk == 'offensive':
            df_factor.rename(columns={'Offensive': 'offensive'}, inplace=True)
        elif fk == 'defensive':
            df_factor.rename(columns={'Defensive_Rating': 'defensive'}, inplace=True)
        elif fk == 'adj_rating':
            df_factor.rename(columns={'Adj_Rating': 'adj_rating'}, inplace=True)
        elif fk == 'efg':
            df_factor.rename(columns={'EFG': 'efg'}, inplace=True)
        elif fk == 'fta':
            df_factor.rename(columns={'FTA_Rate': 'fta'}, inplace=True)
        elif fk == 'turnovers':
            df_factor.rename(columns={'Turnovers': 'turnovers'}, inplace=True)
        # etc...
        
        if merged_df is None:
            # First factor
            merged_df = df_factor
        else:
            # Merge subsequent factors
            merged_df = pd.merge(merged_df, df_factor, on='Name', how='inner')
    
    return merged_df


def load_bracket_results(year: int) -> pd.DataFrame:
    """
    Loads bracket results for a given year and calculates the
    'Tournament Progress' for each team.
    """
    df = pd.read_excel(get_filepath("bracket", year))
    df = calculate_tournament_progress(df)
    return df

###############################################################################
#                           3. BRACKET HELPERS
###############################################################################

def calculate_tournament_progress(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a bracket results DataFrame with columns like
    'First Round Result', 'Second Round Result', etc., 
    determines how far a team advanced (1, 2, 3...).
    """
    rounds = ['First Round', 'Second Round', 'Third Round', 'Fourth Round', 'Fifth Round', 'Sixth Round']
    df['Tournament Progress'] = 0
    for i, round_name in enumerate(rounds, start=1):
        round_result_col = f'{round_name} Result'
        if round_result_col in df.columns:
            df['Tournament Progress'] = df.apply(
                lambda x: i if pd.notnull(x[round_result_col]) and x[round_result_col] == 'Win' 
                else x['Tournament Progress'], 
                axis=1
            )
    return df

def exclude_top_seeds(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Returns a filtered DataFrame that excludes any matchup
    where Team_Seed <= top_n OR Opp_Seed <= top_n.
    """
    return df[
        (df['Pace_Diff'] < top_n) & (df['Pace_Diff'] > -top_n)
    ]


def prepare_bracket_with_factors(bracket_df: pd.DataFrame, factor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges bracket data (TeamName, OpponentName, etc.) with the factor DataFrame
    (which has pace, net rating, etc.), producing a single matchup DataFrame:
       TeamName, Team_Pace, Team_net_rating, ...
       OppName,  Opp_Pace,  Opp_net_rating, ...
    """
    df = bracket_df.copy()

    df.rename(columns={
        "Team Name": "TeamName",
        "First Round Opponent": "OpponentName",
        "Seed": "Team_Seed"
    }, inplace=True)
    
    # Identify winner if you have "First Round Result" etc.
    df["WinnerName"] = np.where(df["First Round Result"] == "Win", df["TeamName"], df["OpponentName"])
    
    # Merge factors for the "TeamName"
    df = pd.merge(
        df,
        factor_df,
        how='left',
        left_on='TeamName',
        right_on='Name'
    ).rename(columns={
        'pace': 'Team_Pace',
        'net_rating': 'Team_NET',
        'rebounds': 'Team_Rebounds',
        'offensive': 'Team_Offensive',
        'defensive': 'Team_Defensive',
        'adj_rating': 'Team_Adj_Rating',
        'efg': 'Team_EFG',
        'fta': 'Team_FTA',
        'turnovers': 'Team_Turnovers',
        # etc. Or keep a loop approach
    })

    # Merge factors for the "OpponentName"
    df = pd.merge(
        df,
        factor_df,
        how='left',
        left_on='OpponentName',
        right_on='Name'
    ).rename(columns={
        'pace': 'Opp_Pace',
        'net_rating': 'Opp_NET',
        'rebounds': 'Opp_Rebounds',
        'offensive': 'Opp_Offensive',
        'defensive': 'Opp_Defensive',
        'adj_rating': 'Opp_Adj_Rating',
        'efg': 'Opp_EFG',
        'fta': 'Opp_FTA',
        'turnovers': 'Opp_Turnovers',
        # etc...
    })

    # Calculate differences if you want them
    df['Pace_Diff'] = df['Team_Pace'] - df['Opp_Pace']
    df['NET_Diff']  = df['Team_NET']  - df['Opp_NET']
    df['Rebounds_Diff'] = df['Team_Rebounds'] - df['Opp_Rebounds']
    df['offense_vs_opp_def'] = df['Team_Offensive'] - df['Opp_Defensive']
    #df['OReb_Diff'] = df['Team_OReb'] - df['Opp_OReb']

    # ... any other transformations ...
    
    # Example for seeds
    df["Opp_Seed"] = df["OpponentName"].map(df.set_index("TeamName")["Team_Seed"])
    # Convert seeds to numeric
    df["Team_Seed"] = pd.to_numeric(df["Team_Seed"], errors="coerce")
    df["Opp_Seed"]  = pd.to_numeric(df["Opp_Seed"], errors="coerce")

    # Exclude rows missing factor data if desired
    factor_cols = ["Team_Pace","Opp_Pace","Team_NET","Opp_NET","Team_Adj_Rating","Opp_Adj_Rating","Team_Rebounds","Opp_Rebounds","Team_Offensive","Opp_Offensive","Team_Defensive","Opp_Defensive"]
    df.dropna(subset=factor_cols, inplace=True)

    return df


###############################################################################
#                   4. CREATE LONG-FORM DATAFRAME FOR PLOTTING
###############################################################################

def create_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a bracket matchup DataFrame (with TeamName, OpponentName, etc.),
    produce a 'long' DataFrame suitable for bar plotting. 
    One row for each role (team vs opponent).
    """
    # Create one row for the "Team"
    team_rows = df[['MatchupID','TeamName','Team_Pace','WinnerName']].copy()
    team_rows['Role'] = 'Team'
    team_rows.rename(columns={
        'TeamName': 'Name',
        'Team_Pace': 'Pace'
    }, inplace=True)

    # Create one row for the "Opponent"
    opp_rows = df[['MatchupID','OpponentName','Opp_Pace','WinnerName']].copy()
    opp_rows['Role'] = 'Opponent'
    opp_rows.rename(columns={
        'OpponentName': 'Name',
        'Opp_Pace': 'Pace'
    }, inplace=True)

    # Concatenate
    long_df = pd.concat([team_rows, opp_rows], ignore_index=True)
    long_df['Won'] = (long_df['Name'] == long_df['WinnerName']).astype(int)

    return long_df

###############################################################################
#                   5. PLOTTING FUNCTION
###############################################################################

def plot_first_round_pace(plot_df: pd.DataFrame, year: int):
    """
    Creates and shows a bar plot using Plotly for the given year's matchup data.
    """
    fig = px.bar(
        plot_df,
        x='MatchupID',
        y='Pace',
        color='Role',          # "Team" vs "Opponent" color
        barmode='group',
        hover_data=['Name'],
        pattern_shape='Won',   # 0 or 1
        pattern_shape_sequence=['x', '']  # 'x' for losers, no pattern for winners
    )

    fig.update_layout(
        title=f'{year} First Round: Team vs Opponent Pace (Winner Highlighted)',
        xaxis_title='Matchup (Index)',
        yaxis_title='Pace',
        legend_title='Role'
    )

    fig.show()

###############################################################################
#                           6. MAIN WORKFLOW
###############################################################################

if __name__ == "__main__":
    # List all years that you have data for
    years = [2021, 2022, 2023, 2024, 2025]

    all_years_rows = []  # hold per-year slices to combine later

    for year in years:
        print(f"Processing year {year}...")

        # 1) Load all factor data (Pace, NET, Rebounds, etc.)
        factor_data = load_factors_for_year(
            year,
            factor_keys=["pace", "net_rating", "rebounds", "offensive", "defensive",
                         "adj_rating", "efg", "fta", "turnovers"]
        )

        # 2) Load bracket data
        bracket_path = DATA_PATHS["bracket"].format(year=year)
        bracket_df = pd.read_excel(bracket_path)

        # 3) Merge bracket + factor data into matchup DataFrame
        df_year = prepare_bracket_with_factors(bracket_df, factor_data)
        print(f"   Total matchups after merges: {len(df_year)}")

        # 4) Identify if 'TeamName' side won
        df_year['Team_Won'] = (df_year['WinnerName'] == df_year['TeamName']).astype(int)

        # 5) Add Year column and select/export columns
        df_year['Year'] = year
        cols = [
            'Year',
            'TeamName', 'OpponentName', 'WinnerName',
            'Team_Pace', 'Opp_Pace',
            'Team_Seed', 'Opp_Seed',
            'NET_Diff', 'Team_NET', 'Opp_NET',
            'Team_Turnovers', 'Opp_Turnovers',
            'Team_EFG', 'Opp_EFG',
            'Team_FTA', 'Opp_FTA',
            'Team_Rebounds', 'Rebounds_Diff',
            'Team_Won'
        ]
        # keep only columns that exist (in case any are missing)
        cols = [c for c in cols if c in df_year.columns]
        all_years_rows.append(df_year[cols])

    # 6) Concatenate all years and write a single Excel file
    if all_years_rows:
        combined = pd.concat(all_years_rows, ignore_index=True)
        out_path = r"C:\Users\888\Desktop\2024 NCAA Stats\df_first_round_all_years.xlsx"
        combined.to_excel(out_path, index=False)
        print(f"Wrote combined dataset to: {out_path}")
    else:
        print("No dataframes were created; nothing to write.")

    print("All processing done!")

