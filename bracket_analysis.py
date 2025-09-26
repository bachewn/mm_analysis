import pandas as pd
import plotly.express as px
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
import numpy as np

###############################################################################
#                           1. CONFIG & HELPERS
###############################################################################

# Adjust this dictionary or generate file paths dynamically:
DATA_PATHS = {
    "pace":       r"C:\Users\888\Desktop\2024 NCAA Stats\rebounds_{year}.xlsx",
    "season_rec": r"C:\Users\888\Desktop\2024 NCAA Stats\season_record_{year}.xlsx",
    "bracket":    r"C:\Users\888\Desktop\2024 NCAA Stats\bracket_results_{year}.xlsx",
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


def prepare_bracket_matchups(bracket_df: pd.DataFrame, combined_df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Cleans, merges, and organizes bracket DataFrame with combined
    pace DataFrame to create a consistent matchup DataFrame for the given year.
    
    Returns a DataFrame with columns like:
      - TeamName, OpponentName, WinnerName
      - Team_Pace, Opp_Pace, Pace_Diff
      - etc.
    """
    # Make a copy so we don't modify the original bracket data
    df = bracket_df.copy()

    # Create a MatchupKey by sorting the two team names alphabetically
    if "Team Name" in df.columns and "First Round Opponent" in df.columns:
        df['MatchupKey'] = df.apply(
            lambda x: tuple(sorted([x['Team Name'], x['First Round Opponent']])),
            axis=1
        )
        # Drop duplicate matchups
        df.drop_duplicates(subset='MatchupKey', inplace=True)

        # Rename columns for clarity
        df.rename(columns={
            'Team Name': 'TeamName',
            'First Round Opponent': 'OpponentName',
            'Seed': 'Team_Seed'
        }, inplace=True)
        
        seed_lookup = df.set_index('TeamName')['Team_Seed']

        # Identify the game winner
        first_round_result_col = 'First Round Result'
        if first_round_result_col in df.columns:
            df['WinnerName'] = np.where(
                df[first_round_result_col] == 'Win',
                df['TeamName'],
                df['OpponentName']
            )

    # Merge in the team's pace
    df = pd.merge(
        df,
        combined_df[['Name','Rebounds']],
        how='left',
        left_on='TeamName',
        right_on='Name'
    ).rename(columns={'Rebounds': 'Team_Pace'})

    missing_team_rows = df[df['Team_Pace'].isna()]
    print("\nTeams with no pace data (TeamName):")
    print(missing_team_rows[['TeamName','OpponentName']])

    # Merge in the opponent's pace
    df = pd.merge(
        df,
        combined_df[['Name','Rebounds']],
        how='left',
        left_on='OpponentName',
        right_on='Name'
    ).rename(columns={'Rebounds': 'Opp_Pace'})
    missing_team_rows = df[df['Opp_Pace'].isna()]
    print("\nTeams with no pace data (OpponentName):")
    print(missing_team_rows[['TeamName','OpponentName']])

    # Compute pace difference
    df['Pace_Diff'] = df['Team_Pace'] - df['Opp_Pace']

    # Drop rows with missing pace data
    df.dropna(subset=['Team_Pace','Opp_Pace'], inplace=True)

    # -- Now attach seeds for each side --
    df['Team_Seed'] = df['TeamName'].map(seed_lookup)
    df['Opp_Seed']  = df['OpponentName'].map(seed_lookup)

    # Convert to numeric if necessary
    df['Team_Seed'] = pd.to_numeric(df['Team_Seed'], errors='coerce')
    df['Opp_Seed']  = pd.to_numeric(df['Opp_Seed'], errors='coerce')

    # Create a unique MatchupID
    df['MatchupID'] = range(1, len(df) + 1)

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
    years = [2021,2022, 2023, 2024, 2025]  # You can just append 2024, 2025, etc.

    for y in years:
        print(f"Processing year {y}...")

        # 1. Load and prepare season (pace + W/L) data
        season_data = load_and_prepare_season_data(y)
        print(f"  After loading season_data: {len(season_data)} rows")

        # 2. Load bracket results
        bracket_data = load_bracket_results(y)
        print(f"  After loading bracket_data: {len(bracket_data)} rows")

        # 3. Merge bracket data with season_data
        df_year = prepare_bracket_matchups(bracket_data, season_data, y)
        print(f"  After prepare_bracket_matchups: {len(df_year)} rows")

        df_year['Team_Won'] = (df_year['WinnerName'] == df_year['TeamName'])
        # Check if the TeamName side had the higher pace
        df_year['Team_Higher_Pace'] = (df_year['Team_Pace'] > df_year['Opp_Pace'])

        # We want a boolean: "did the winner have the higher pace?"
        # If Team_Won is True, winner had higher pace if Team_Higher_Pace is True.
        # If Team_Won is False, winner had higher pace if Opp_Pace > Team_Pace.
        df_year['Winner_Had_Higher_Pace'] = df_year.apply(
            lambda row: (row['Team_Won']  and row['Team_Higher_Pace']) or
                        (not row['Team_Won'] and not row['Team_Higher_Pace']),
            axis=1
        )

        pct_higher_pace = df_year['Winner_Had_Higher_Pace'].mean() * 100
        print(f"  {pct_higher_pace:.1f}% of matchups were won by the team with the higher pace in {y}.")

        df_year['Team_Seed'] = pd.to_numeric(df_year['Team_Seed'], errors='coerce')
        df_year['Opp_Seed']  = pd.to_numeric(df_year['Opp_Seed'], errors='coerce')

        print("Number of rows before filtering:", len(df_year))

        # Exclude matchups if EITHER side is seed 1,2,3
        df_filtered = df_year[
            (df_year['Team_Seed'] > 3)
        ]

        print("Number of rows after filtering:", len(df_filtered))

        print("Number of rows after filtering:", df_filtered)

        pct_filtered = df_filtered['Winner_Had_Higher_Pace'].mean() * 100
        print(f"  Ignoring seeds 1-3: {pct_filtered:.1f}% of winners had the higher pace in {y}.")

        # 4. Create a long-form DataFrame for bar plotting
       # plot_df_year = create_plot_df(df_year)
        #print(f"  Final plot_df_{y} has: {len(plot_df_year)} rows")

        # 5. Plot
        #plot_first_round_pace(plot_df_year, y)


    print("All processing done!")
