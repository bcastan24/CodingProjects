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
START = 2002
END = 2004

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
scaler = MinMaxScaler()
#batting.loc[:, selectedColumns] = scaler.fit_transform(batting[selectedColumns])
batting[selectedColumns] = scaler.fit_transform(batting[selectedColumns].astype("float32"))
sfs.fit(batting[selectedColumns], batting["Next_WAR"])
predictors = list(selectedColumns[sfs.get_support()])

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
mean_squared_error(predictions["actual"], predictions["prediction"])



