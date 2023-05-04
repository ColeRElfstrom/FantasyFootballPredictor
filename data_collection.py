import nfl_data_py as nfl
import pandas as pd

df_2022 = nfl.import_weekly_data([2022], columns=['week', 'player_id', 'passing_yards', 'passing_tds', 'interceptions', 'rushing_yards',
       'rushing_tds', 'rushing_fumbles_lost', 'rushing_2pt_conversions', 'receptions', 'receiving_yards', 'receiving_tds', 
       'receiving_fumbles_lost', 'receiving_2pt_conversions', 'special_teams_tds', 'fantasy_points_ppr'])

df_roster = nfl.import_rosters([2022], columns=["player_id", "position", "player_name", "team"])
df_2022 = df_2022.merge(df_roster, on="player_id", how="inner")


sch_df = nfl.import_schedules([2022])


sch_df = sch_df[["away_team", "week", "home_team", "temp", "wind"]]

home_df = pd.merge(df_2022, sch_df, left_on=["week", "team"], right_on=["week", "home_team"], how="right")

away_df = pd.merge(df_2022, sch_df, left_on=["week", "team"], right_on=["week", "away_team"], how="right")

full_df = pd.concat([home_df, away_df])

opp_df = []
for index, row in full_df.iterrows():
       if row.away_team == row.team:
              opp_df.append(row.home_team)
       else:
              opp_df.append(row.away_team)


full_df["Opponent"] = opp_df

full_df = full_df.drop(columns=["away_team", "home_team"])


def_df = pd.read_csv("./defense.csv")

def_df = def_df.rename(columns={"P_Y/G": "pypg", "R_Y/G": "rypg"})

for index, row in def_df.iterrows():
       row.pypg = row.pypg / def_df.iloc[32]["pypg"]
       row.rypg = row.rypg / def_df.iloc[32]["rypg"]
       def_df.iloc[index] = row

full_df = full_df.merge(def_df, left_on=["Opponent"], right_on=["Tm"], how="inner")

full_df = full_df.drop(columns=["Tm"])

from google.cloud import storage


fpvd_df = full_df[["player_name", "fantasy_points_ppr", "Opponent", "rypg", "pypg"]]
train_df = full_df[["player_name", "fantasy_points_ppr", "rypg", "pypg", "position", "week", "temp", "wind" ]]


uniques = fpvd_df["Opponent"].unique()


fp_teams = pd.DataFrame(uniques, columns=["team"])

scores_sum = []

from matplotlib import pyplot as plt

def return_full_df():
       uniques = train_df["player_name"].unique()
       temp_unique = pd.DataFrame(uniques, columns=["player_name"])
       return temp_unique

       
def get_player_data(name):
    temp_df = train_df.loc[train_df["player_name"] == name].copy()
    mean = temp_df["fantasy_points_ppr"].mean()
    temp_df.loc[:, "avg_ppg"] = mean
    if('RB' in temp_df['position']):
       temp_df["ppg"] = temp_df["rypg"]
    else:
       temp_df["ppg"] = temp_df["pypg"]
    return temp_df

def get_all_player_data():
       uniques = train_df["player_name"].unique()
       ret_df = pd.DataFrame()
       for name in uniques:
              temp_df = get_player_data(name)
              ret_df = pd.concat([ret_df, temp_df])
       return ret_df




from sklearn.model_selection import train_test_split
 
# Read data
data = get_all_player_data()
data = data[["fantasy_points_ppr", "ppg", "avg_ppg"]]
data.fillna(0)
X, y = data.drop(columns="fantasy_points_ppr").values.tolist(), data["fantasy_points_ppr"].tolist()

# train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True)
 
import numpy as np
from sklearn.linear_model import LinearRegression


reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

def train(player):
       data = get_player_data(player)
       data = data[["fantasy_points_ppr", "ppg", "avg_ppg"]]
       data.fillna(0)
       X, y = data.drop(columns="fantasy_points_ppr").values.tolist(), data["fantasy_points_ppr"].tolist()

       
       # train-test split for model evaluation
       X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.95, shuffle=True)

       #reg.fit(X_train, y_train)

       y_pred = reg.predict(X_test)

       return y_pred, y_test

def plot_player(player):
       data = get_player_data(player)
       data = data[["fantasy_points_ppr", "ppg", "avg_ppg", "week"]]
       data.fillna(0)
       X, y = data["week"], data["fantasy_points_ppr"]
       a, b = np.polyfit(X, y, 1)
       fig = plt.figure()
       plt.scatter(X, y)
       plt.plot(X, a*X+b)
       plt.title(player)
       plt.xlabel("Week")
       plt.ylabel("Fantasy Points")
       return fig
