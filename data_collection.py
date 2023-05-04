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

def_df = def_df[["Tm", "Pts/G"]]
def_df = def_df.rename(columns={"Pts/G": "ppg"})

for index, row in def_df.iterrows():
       row.ppg = row.ppg / def_df.iloc[32]["ppg"]
       def_df.iloc[index] = row

full_df = full_df.merge(def_df, left_on=["Opponent"], right_on=["Tm"], how="inner")

full_df = full_df.drop(columns=["Tm"])

from google.cloud import storage

# path_to_private_key = './gcp_key.json'
# client = storage.Client.from_service_account_json(json_credentials_path=path_to_private_key)

# # The NEW bucket on GCS in which to write the CSV file
# bucket = client.bucket('demo-bucket-325ce')
# # The name assigned to the CSV file on GCS
# blob = bucket.blob('test-data.csv')
# blob.upload_from_string(full_df.to_csv(), 'text/csv')




fpvd_df = full_df[["player_name", "fantasy_points_ppr", "Opponent", "ppg"]]
train_df = full_df[["player_name", "fantasy_points_ppr", "ppg", "position", "week", "temp", "wind" ]]


uniques = fpvd_df["Opponent"].unique()


fp_teams = pd.DataFrame(uniques, columns=["team"])

scores_sum = []

# for index, team in fp_teams.iterrows():
#        scores_sum.append(0)
#        for p_index, player in fpvd_df.iterrows():
#               if team.team == player.Opponent:
#                     scores_sum[index] += player.fantasy_points_ppr

# avg = sum(scores_sum) / 32


# avg_scores = []

# for team in scores_sum:
#        avg_scores.append(team / avg)

# fp_teams["score"] = avg_scores


# fpvd_df = fpvd_df.merge(fp_teams, left_on="Opponent", right_on="team", how="inner")

# fpvd_df = fpvd_df.drop(columns=["team"])

# def_df = def_df.merge(fp_teams, left_on="Tm", right_on="team", how="inner")
# def_df = def_df.drop(columns=["team"])

from matplotlib import pyplot as plt

# corr = def_df["ppg"].corr(def_df["score"])


# plt.plot(def_df["ppg"], def_df["score"])
# plt.xlabel("Defense PPG Allowed AVG")
# plt.ylabel("Defense Fantasy Points Allowed AVG")
# plt.show()


# temp_df = full_df[["temp", "fantasy_points_ppr"]]

# temp_df = temp_df.dropna()

# uniques = temp_df["temp"].unique()

# temp_unique = pd.DataFrame(uniques, columns=["temp"])


# temp_sum = []
# temp_count = []

# for index, temp in temp_unique.iterrows():
#        temp_sum.append(0)
#        temp_count.append(0)
#        for p_index, player in temp_df.iterrows():
#               if temp.temp == player.temp:
#                     temp_sum[index] += player.fantasy_points_ppr
#                     temp_count[index] += 1


# temp_scores = []

# for index in range(len(temp_count)):
#        temp_scores.append(temp_sum[index] / temp_count[index])

# temp_unique["scores"] = temp_scores


# corr = temp_unique["scores"].corr(temp_unique["temp"])

# temp_unique = temp_unique.sort_values("temp")

# # plt.plot(temp_unique["temp"], temp_unique["scores"])
# # plt.xlabel("Temperature (F)")
# # plt.ylabel("Fantasy Points")
# # plt.show()

def return_full_df():
       uniques = train_df["player_name"].unique()
       temp_unique = pd.DataFrame(uniques, columns=["player_name"])
       return temp_unique

       
def get_player_data(name):
    temp_df = train_df.loc[train_df["player_name"] == name].copy()
    mean = temp_df["fantasy_points_ppr"].mean()
    temp_df.loc[:, "avg_ppg"] = mean
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
