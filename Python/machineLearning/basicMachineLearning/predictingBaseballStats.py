#An algorithm to predict a baseball player's WAR
import os
import pandas as pd
import numpy as np
from pybaseball import batting_stats
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#defining the start and end year that we want to pull data from
START = 2014
END = 2024

#downloading the data
batting = batting_stats(START, END, qual=200)
batting.to_csv("batting.csv")
#splitting data into groups based on player id and removing players that we only have one season of data on
batting = batting.groupby("IDfg", as_index=False).filter(lambda x: x.shape[0] > 1)

#Machine Learning target
def nextSeason(player):
    player = player.sort_values("Season")
    player["Next_WAR"] = player["WAR"].shift(-1)
    return player

batting = batting.groupby("IDfg").apply(nextSeason, include_groups=False)

#Cleaning Data
#Counting all the columns that are empty
nullCount = batting.isnull().sum()
#selecting all the columns that are complete
completeCols = list(batting.columns[nullCount == 0])
#Removing all of the null columns
batting = batting[completeCols + ["Next_WAR"]].copy()
#Because our machine learning only takes numbers we need to get rid of the columns that are strings or turn them into numbers
#Deleting columns that are strings that we don't need
del batting["Dol"]
del batting["Age Rng"]
#turning team name into a number
batting["team_code"] = batting["Team"].astype("category").cat.codes
#making a copy of the data
batting_full = batting.copy()
batting = batting.dropna().copy()

#selecting useful features
rr = Ridge(alpha=1)
#split data into 3 parts but make sure that ML is not using data from the future to predict WARs of the past
split = TimeSeriesSplit(n_splits=3)
sfs = SequentialFeatureSelector(rr, n_features_to_select=20, direction="forward", cv=split, n_jobs=4)
#taking out columns that we don't need to run through ML
removedColumns = ["Next_WAR", "Name", "Team", "IDfg", "Season"]
#take all the columns that are not in removedColumns
selectedColumns = batting.columns[~batting.columns.isin(removedColumns)]
#makes it so every number is between 1 and 0
scaler = MinMaxScaler()
batting[selectedColumns] = scaler.fit_transform(batting[selectedColumns].astype("float32"))
sfs.fit(batting[selectedColumns], batting["Next_WAR"])
predictors = list(selectedColumns[sfs.get_support()])

#making the computer make predictions of next seasons WAR
def backtest(data, model, predictors, start=5, step=1):
    allPredictions = []
    years = sorted(data["Season"].unique())
    #each time through this loop we are going to use historical data to predict a single season
    for i in range(start, len(years), step):
        currentYear = years[i]
        #out of all the years of data we have we will set a current year somewhere a couple years into the first year that we have and then we are going to use all the data from before the "current year" that we set to train the model
        train = data[data["Season"] < currentYear]
        #then we are going to use the trained data to predict the WAR for the "current year" and everytime we loop thru this loop we set the "current year" to the next year and in our data set until we have predictions for all the years past the first "current year" that we set
        test = data[data["Season"] == currentYear]
        model.fit(train[predictors], train["Next_WAR"])
        preds = model.predict(test[predictors])
        #this returns a numpy array but we're going to set it to a pandas series
        preds = pd.Series(preds, index=test.index)
        #combine actual next seasons WAR with our prediction
        combined = pd.concat([test["Next_WAR"], preds], axis=1)
        #name the columns
        combined.columns = ["actual", "prediction"]
        #add the columns to allPredictions
        allPredictions.append(combined)

    return pd.concat(allPredictions)

predictions = backtest(batting, rr, predictors)
#calculating our error
mean_squared_error(predictions["actual"], predictions["prediction"])

#allowing the computer to use past player history to make more informed predictions about the next seasons WAR, with just back testing the algorithm can only make predictions based on the current years data
def playerHistory(df):
    df = df.sort_values("Season")
    #number to indicate what season this is for a player, ie their first season, second season
    df["player_season"] = range(0, df.shape[0])
    #compute WAR correlation
    df["war_corr"] = list(df[["player_season","WAR"]].expanding().corr().loc[(slice(None), "player_season"), "WAR"])
    df["war_corr"].fillna(1, inplace=True)
    #compute WAR difference, takes WAR from current season divided by WAR from previous season
    df["war_diff"] = df["WAR"] / df["WAR"].shift(1)
    df["war_diff"].fillna(1, inplace=True)
    #any values that have infinite values get replaced with a 1
    df["war_diff"][df["war_diff"] == np.inf] = 1
    return df

#grouping by player and apply player history
batting = batting.groupby("IDfg", group_keys=False).apply(playerHistory)

#find averages across a whole season and compare to how player did
def groupAverages(df):
    return df["WAR"] / df["WAR"].mean()

#find difference between how this player did in a season vs how the average player did in this season
batting["war_season"] = batting.groupby("Season", group_keys=False).apply(groupAverages)

#making a list of predictors that we already had and all the new ones we added
newPredictors = predictors + ["player_season", "war_corr", "war_season", "war_diff"]

#making predictions
predictions = backtest(batting, rr, newPredictors)

#I'm going to sort the predictions in a way that make it easier to read
merged = predictions.merge(batting, left_index=True, right_index=True)
merged["diff"] = (predictions["actual"] - predictions["prediction"]).abs()
lastMerged = merged[["IDfg", "Season", "Name", "WAR", "Next_WAR", "diff"]]
lastMerged.to_csv("finalPredictionStats.csv")



