import pandas as pd
import plotly.express as px
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression

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

# B. Create a binary 'Won' column: 1 if team won the first round, 0 if not
df_2022['Won'] = (df_2022['First Round Result'] == 'Win').astype(int)

# C. Merge to get the *team's* pace
df_2022 = pd.merge(
    df_2022,
    combined_2022[['Name', 'Pace']],
    how='left',
    left_on='Team Name',
    right_on='Name'
)
df_2022.rename(columns={'Pace': 'Team_Pace'}, inplace=True)

# D. Merge again to get the *opponent's* pace
df_2022 = pd.merge(
    df_2022,
    combined_2022[['Name', 'Pace']],
    how='left',
    left_on='First Round Opponent',
    right_on='Name'
)
df_2022.rename(columns={'Pace': 'Opp_Pace'}, inplace=True)

# E. Calculate pace difference: Team_Pace - Opp_Pace
df_2022['Pace_Diff'] = df_2022['Team_Pace'] - df_2022['Opp_Pace']

# F. Drop any rows missing data
df_2022 = df_2022.dropna(subset=['Pace_Diff', 'Won'])

# --- 1) Point-biserial correlation ---
corr_2022, pval_2022 = pointbiserialr(df_2022['Won'], df_2022['Pace_Diff'])

print("=== 2022 Analysis ===")
print(f"Number of first-round matchups found: {len(df_2022)}")
print(f"Point-Biserial Correlation (Won ~ Pace_Diff) = {corr_2022:.4f}")
print(f"p-value = {pval_2022:.4f}")

# --- 2) Logistic Regression ---
X_2022 = df_2022[['Pace_Diff']]
y_2022 = df_2022['Won']

logreg_2022 = LogisticRegression()
logreg_2022.fit(X_2022, y_2022)

coef_2022 = logreg_2022.coef_[0][0]
intercept_2022 = logreg_2022.intercept_[0]
print(f"Logistic Regression Coefficient (Pace_Diff): {coef_2022:.4f}")
print(f"Logistic Regression Intercept: {intercept_2022:.4f}")

# (Optional) Example probability of winning when Pace_Diff = +5
prob_win_if_diff_5 = logreg_2022.predict_proba([[5]])[0][1]
print(f"Predicted prob. of winning when Team_Pace is 5 pts higher than Opp. = {prob_win_if_diff_5:.3f}")
print("")

# --------------------
# ----  FOR 2023  ----
# --------------------

df_2023 = bracket_results_2023.copy()
df_2023['Won'] = (df_2023['First Round Result'] == 'Win').astype(int)

# Merge for team pace
df_2023 = pd.merge(
    df_2023,
    combined_2023[['Name', 'Pace']],
    how='left',
    left_on='Team Name',
    right_on='Name'
)
df_2023.rename(columns={'Pace': 'Team_Pace'}, inplace=True)

# Merge for opponent pace
df_2023 = pd.merge(
    df_2023,
    combined_2023[['Name', 'Pace']],
    how='left',
    left_on='First Round Opponent',
    right_on='Name'
)
df_2023.rename(columns={'Pace': 'Opp_Pace'}, inplace=True)

df_2023['Pace_Diff'] = df_2023['Team_Pace'] - df_2023['Opp_Pace']
df_2023 = df_2023.dropna(subset=['Pace_Diff', 'Won'])

# --- 1) Point-biserial correlation ---
corr_2023, pval_2023 = pointbiserialr(df_2023['Won'], df_2023['Pace_Diff'])

print("=== 2023 Analysis ===")
print(f"Number of first-round matchups found: {len(df_2023)}")
print(f"Point-Biserial Correlation (Won ~ Pace_Diff) = {corr_2023:.4f}")
print(f"p-value = {pval_2023:.4f}")

# --- 2) Logistic Regression ---
X_2023 = df_2023[['Pace_Diff']]
y_2023 = df_2023['Won']

logreg_2023 = LogisticRegression()
logreg_2023.fit(X_2023, y_2023)

coef_2023 = logreg_2023.coef_[0][0]
intercept_2023 = logreg_2023.intercept_[0]
print(f"Logistic Regression Coefficient (Pace_Diff): {coef_2023:.4f}")
print(f"Logistic Regression Intercept: {intercept_2023:.4f}")

prob_win_if_diff_5 = logreg_2023.predict_proba([[5]])[0][1]
print(f"Predicted prob. of winning with Pace_Diff=+5: {prob_win_if_diff_5:.3f}")