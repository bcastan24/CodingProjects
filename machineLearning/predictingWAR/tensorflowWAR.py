import pandas as pd
import numpy as np
from pybaseball import batting_stats
import tensorflow as tf
import keras
from keras import layers

#Gathering and cleaning the data
#Note I copied this code from predictingStatsUsingNN.py in the basicMachineLearning folder
#defining the start and end year that we want to pull data from
START = 2015
END = 2024

#downloading the data
batting = batting_stats(START, END, qual=200)
batting.to_csv("batting.csv")
#splitting data into groups based on player id and removing players that have less than 4 seasons of data on
batting = batting.groupby("IDfg", as_index=False).filter(lambda x: x.shape[0] > 4)

#Machine Learning target
def nextSeason(player):
    player = player.sort_values("Season")
    player["Next_WAR"] = player["WAR"].shift(-1)
    return player

batting = batting.groupby("IDfg").apply(nextSeason, include_groups=False)

#remove rows where Next_WAR is NaN
batting = batting.dropna(subset=['Next_WAR'])

#Cleaning Data
#selecting the columns that we need to run through NN
selectedColumns = ["WAR", "OPS", "BABIP", "wRC+"]
targetColumn = "Next_WAR"

#check if we have the required columns
missingCols = [col for col in selectedColumns + [targetColumn] if col not in batting.columns]
if missingCols:
    print(f"Missing Columns: {missingCols}")
    print(f"Available Columns: {batting.columns.tolist()}")

#remove any rows with NaN in selected columns
batting = batting.dropna(subset=selectedColumns + [targetColumn])

#convert to float
for col in selectedColumns + [targetColumn]:
    batting[col] = pd.to_numeric(batting[col], errors='coerce')

#remove rows that become NaN after conversion
batting = batting.dropna(subset=selectedColumns + [targetColumn])

#split data, 80% used for training, 20% used for testing
split = int(len(batting) * 0.8)
trainingData = batting.iloc[:split]
testingData = batting.iloc[split:]


#get selected data and convert to numpy arrays
trainSelected = trainingData[selectedColumns].values
testSelected = testingData[selectedColumns].values
trainTargets = trainingData[targetColumn].values.reshape(-1,1)
testTargets = testingData[targetColumn].values.reshape(-1,1)

#create training sets
def createTrainingSets(selected, targets, sequenceLength = 4):
    X = []
    y = []
    for i in range(0, len(selected) - sequenceLength + 1, 5):
        if i + sequenceLength < len(selected):
            #make the first four years of selectedColumns into one vector
            sequence = selected[i:i+sequenceLength].flatten()
            #get the fifth season WAR
            target = targets[i+sequenceLength]
            X.append(sequence)
            y.append(target)
    return np.array(X), np.array(y).flatten()

#create normalizer layer
normalizer = keras.layers.Normalization(axis=-1)

