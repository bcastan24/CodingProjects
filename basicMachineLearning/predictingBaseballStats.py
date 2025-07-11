#An algorithm to predict a baseball player's WAR
import os
import pandas as pd
import numpy as np
from pybaseball import batting_stats

#defining the start and end year that we want to pull data from
START = 2002
END = 2022

#downloading the data
batting = batting_stats(START, END, qual=200)
batting.to_csv("batting.csv")
#splitting data into groups based on player id and removing players that we only have one season of data on
batting = batting.groupby("IDfg", group_keys=False).filter(lambda x: x.shape[0] > 1)


