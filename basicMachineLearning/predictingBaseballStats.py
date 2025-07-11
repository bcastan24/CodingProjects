#An algorithm to predict a baseball player's WAR
import os
import pandas as pd
import numpy as np
from pybaseball import batting_stats

#defining the start and end year that we want to pull data from
START = 2002
END = 2004

#downloading the data
batting = batting_stats(START, END, qual=200)
batting.to_csv("batting.csv")
#splitting data into groups based on player id and removing players that we only have one season of data on
batting = batting.groupby("IDfg").filter(lambda x: x.shape[0] > 1)

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
batting = batting.dropna()




