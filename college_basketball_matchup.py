import pandas as pd
import plotly.express as px
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
import numpy as np


# Load data from Excel files
pace_2022 = pd.read_excel(r'C:\Users\888\Desktop\2024 NCAA Stats\pace_2022.xlsx')
wins_losses_2022 = pd.read_excel(r'C:\Users\888\Desktop\2024 NCAA Stats\season_record_2022.xlsx')
pace_2023 = pd.read_excel(r'C:\Users\888\Desktop\2024 NCAA Stats\pace_2023.xlsx')
wins_losses_2023 = pd.read_excel(r'C:\Users\888\Desktop\2024 NCAA Stats\season_record_2023.xlsx')


# Convert wins and losses to numeric to avoid type issues
wins_losses_2022['wins'] = pd.to_numeric(wins_losses_2022['Wins'], errors='coerce')
wins_losses_2022['losses'] = pd.to_numeric(wins_losses_2022['Losses'], errors='coerce')
wins_losses_2023['wins'] = pd.to_numeric(wins_losses_2023['Wins'], errors='coerce')
wins_losses_2023['losses'] = pd.to_numeric(wins_losses_2023['Losses'], errors='coerce')

# Merge pace data with wins-losses data
combined_2022 = pd.merge(wins_losses_2022, pace_2022, on=['Name'], how='inner')
combined_2023 = pd.merge(wins_losses_2023, pace_2023, on=['Name'], how='inner')

# Calculate win-loss ratio
combined_2022['Win-Loss Ratio'] = combined_2022['wins'] / (combined_2022['wins'] + combined_2022['losses'])
combined_2023['Win-Loss Ratio'] = combined_2023['wins'] / (combined_2023['wins'] + combined_2023['losses'])

# Load NCAA tournament results for 2022 and 2023
tournament_2022 = pd.read_excel(r'C:\Users\888\Desktop\2024 NCAA Stats\bracket_results_22.xlsx')
tournament_2023 = pd.read_excel(r'C:\Users\888\Desktop\2024 NCAA Stats\bracket_results_23.xlsx')

# Function to determine how far a team advanced in the tournament
def calculate_tournament_progress(df):
    rounds = ['First Round', 'Second Round', 'Third Round', 'Fourth Round', 'Fifth Round', 'Sixth Round']
    df['Tournament Progress'] = 0
    for i, round_name in enumerate(rounds, start=1):
        round_result_col = f'{round_name} Result'
        df['Tournament Progress'] = df.apply(lambda x: i if pd.notnull(x[round_result_col]) and x[round_result_col] == 'Win' else x['Tournament Progress'], axis=1)
    return df

# Apply the function to both tournament datasets
bracket_results_2022 = calculate_tournament_progress(tournament_2022)
bracket_results_2023 = calculate_tournament_progress(tournament_2023)

# A. Make a copy so we don't modify the original DataFrame
df_2022 = bracket_results_2022.copy()
df_2023 = bracket_results_2023.copy()

# Create a "MatchupKey" by sorting the two team names alphabetically
df_2022['MatchupKey'] = df_2022.apply(
    lambda x: tuple(sorted([x['Team Name'], x['First Round Opponent']])),
    axis=1
)

# Drop duplicate matchups, keeping the first occurrence
df_2022.drop_duplicates(subset='MatchupKey', inplace=True)

df_2023['MatchupKey'] = df_2023.apply(
    lambda x: tuple(sorted([x['Team Name'], x['First Round Opponent']])),
    axis=1
)
df_2023.drop_duplicates(subset='MatchupKey', inplace=True)

# Rename columns for clarity
df_2022.rename(columns={
    'Team Name': 'TeamName',
    'First Round Opponent': 'OpponentName'
}, inplace=True)

df_2022['WinnerName'] = np.where(
    df_2022['First Round Result'] == 'Win',
    df_2022['TeamName'],
    df_2022['OpponentName']
)

df_2023.rename(columns={
    'Team Name': 'TeamName',
    'First Round Opponent': 'OpponentName'
}, inplace=True)

df_2023['WinnerName'] = np.where(
    df_2023['First Round Result'] == 'Win',
    df_2023['TeamName'],
    df_2023['OpponentName']
)

# Merge to get the team’s pace
df_2022 = pd.merge(
    df_2022,
    combined_2022[['Name', 'Pace']],
    how='left',
    left_on='TeamName',
    right_on='Name'
).rename(columns={'Pace':'Team_Pace'})

df_2023 = pd.merge(
    df_2023,
    combined_2023[['Name', 'Pace']],
    how='left',
    left_on='TeamName',
    right_on='Name'
).rename(columns={'Pace':'Team_Pace'})

# Merge again to get the opponent’s pace
df_2022 = pd.merge(
    df_2022,
    combined_2022[['Name', 'Pace']],
    how='left',
    left_on='OpponentName',
    right_on='Name'
).rename(columns={'Pace':'Opp_Pace'})

df_2023 = pd.merge(
    df_2023,
    combined_2023[['Name', 'Pace']],
    how='left',
    left_on='OpponentName',
    right_on='Name'
).rename(columns={'Pace':'Opp_Pace'})

# Compute pace difference
df_2022['Pace_Diff'] = df_2022['Team_Pace'] - df_2022['Opp_Pace']
df_2023['Pace_Diff'] = df_2023['Team_Pace'] - df_2023['Opp_Pace']


# Drop rows with missing pace data
df_2022.dropna(subset=['Team_Pace','Opp_Pace'], inplace=True)
df_2023.dropna(subset=['Team_Pace','Opp_Pace'], inplace=True)


# Optional: create a MatchupID so we can group each pair of teams
df_2022['MatchupID'] = range(1, len(df_2022)+1)
df_2023['MatchupID'] = range(1, len(df_2023)+1)


# Create one row for the team
team_rows_2022 = df_2022[['MatchupID','TeamName','Team_Pace','WinnerName']].copy()
team_rows_2022['Role'] = 'Team'
team_rows_2022.rename(columns={
    'TeamName': 'Name',
    'Team_Pace': 'Pace'
}, inplace=True)

# Create one row for the opponent
opp_rows_2022 = df_2022[['MatchupID','OpponentName','Opp_Pace','WinnerName']].copy()
opp_rows_2022['Role'] = 'Opponent'
opp_rows_2022.rename(columns={
    'OpponentName': 'Name',
    'Opp_Pace': 'Pace'
}, inplace=True)

# Concatenate to get "long" format
plot_df_2022 = pd.concat([team_rows_2022, opp_rows_2022], ignore_index=True)
plot_df_2022['Won'] = (plot_df_2022['Name'] == plot_df_2022['WinnerName']).astype(int)


# Repeat similarly for 2023:
team_rows_2023 = df_2023[['MatchupID','TeamName','Team_Pace','WinnerName']].copy()
team_rows_2023['Role'] = 'Team'
team_rows_2023.rename(columns={
    'TeamName': 'Name',
    'Team_Pace': 'Pace'
}, inplace=True)

opp_rows_2023 = df_2023[['MatchupID','OpponentName','Opp_Pace','WinnerName']].copy()
opp_rows_2023['Role'] = 'Opponent'
opp_rows_2023.rename(columns={
    'OpponentName': 'Name',
    'Opp_Pace': 'Pace'
}, inplace=True)

plot_df_2023 = pd.concat([team_rows_2023, opp_rows_2023], ignore_index=True)
plot_df_2023['Won'] = (plot_df_2023['Name'] == plot_df_2023['WinnerName']).astype(int)


fig_2022 = px.bar(
    plot_df_2022,
    x='MatchupID',
    y='Pace',
    color='Role',          # "Team" vs "Opponent" color
    barmode='group',
    hover_data=['Name'],
    pattern_shape='Won',   # 0 or 1
    # You can also customize the pattern shapes:
    pattern_shape_sequence=['x', '']  # 'x' for losers, no pattern for winners
)

fig_2022.update_layout(
    title='2022 First Round: Team vs Opponent Pace (Winner Highlighted)',
    xaxis_title='Matchup (Index)',
    yaxis_title='Pace',
    legend_title='Role'
)

fig_2022.show()


fig_2023 = px.bar(
    plot_df_2023,
    x='MatchupID',
    y='Pace',
    color='Role',          # "Team" vs "Opponent" color
    barmode='group',
    hover_data=['Name'],
    pattern_shape='Won',   # 0 or 1
    # You can also customize the pattern shapes:
    pattern_shape_sequence=['x', '']  # 'x' for losers, no pattern for winners
)

fig_2023.update_layout(
    title='2023First Round: Team vs Opponent Pace (Winner Highlighted)',
    xaxis_title='Matchup (Index)',
    yaxis_title='Pace',
    legend_title='Role'
)

fig_2023.show()
