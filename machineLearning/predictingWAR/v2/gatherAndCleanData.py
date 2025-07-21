import pandas as pd
import numpy as np
from pybaseball import batting_stats
import os

class Data:
    def __init__(self):
        #check if there is a csv file with all the raw data
        self.dataAvail = os.path.exists("batting.csv")
        #if not then make one
        if self.dataAvail == False:
            self.createFile(2015, 2024)

    def createFile(self, start:int, end:int):
        START = start
        END = end

        # downloading the data
        batting = batting_stats(START, END, qual=200)
        batting.to_csv("batting.csv")

    def cleanData(self):
        batting = pd.read_csv("batting.csv")
        batting = batting.groupby("IDfg", as_index=False).filter(lambda x: x.shape[0] > 4)
        def nextSeason(player):
            player = player.sort_values("Season")
            player["Next_WAR"] = player["WAR"].shift(-1)
            return player

        batting = batting.groupby("IDfg").apply(nextSeason, include_groups=False).reset_index()

        # remove rows where Next_WAR is NaN
        batting = batting.dropna(subset=['Next_WAR'])

        # Cleaning Data
        # selecting the columns that we need to run through NN
        selectedColumns = ["Age", "WAR", "wOBA", "EV", "BABIP+", "OBP+", "wRC+"]
        targetColumn = "Next_WAR"

        # check if we have the required columns
        missingCols = [col for col in selectedColumns + [targetColumn] if col not in batting.columns]
        if missingCols:
            print(f"Missing Columns: {missingCols}")
            print(f"Available Columns: {batting.columns.tolist()}")

        # remove any rows with NaN in selected columns
        batting = batting.dropna(subset=selectedColumns + [targetColumn])

        # convert to float
        for col in selectedColumns + [targetColumn]:
            batting[col] = pd.to_numeric(batting[col], errors='coerce')

        # remove rows that become NaN after conversion
        batting = batting.dropna(subset=selectedColumns + [targetColumn])

        # split data, 80% used for training, 20% used for testing
        split = int(len(batting) * 0.8)
        trainingData = batting.iloc[:split]
        testingData = batting.iloc[split:]

        # get selected data and convert to numpy arrays
        trainSelected = trainingData[selectedColumns].values
        testSelected = testingData[selectedColumns].values
        trainTargets = trainingData[targetColumn].values
        testTargets = testingData[targetColumn].values

        return trainSelected, trainTargets, testSelected, testTargets

    def createTrainingSets(self,sequenceLength=4):
        selected, targets, xTest, yTest = self.cleanData()
        X = []
        y = []
        a = []
        b = []
        for i in range(len(selected) - sequenceLength + 1):
            sequence = selected[i:i + sequenceLength].flatten()
            #normalizing values so they hover around 0, for some reason the switch case was giving me errors
            #so I just put a bunch of elif statements instead
            for j in range(len(sequence)):
                if (j == 0 or j % 7 == 0):
                    sequence[j] -= 29.0
                elif (j == 1 or j == 8 or j == 15 or j == 22):
                    sequence[j] -= 2.0
                elif (j == 2 or j == 9 or j == 16 or j == 23):
                    sequence[j] = (sequence[j] * 10) - 3.2
                elif (j == 3 or j == 10 or j == 17 or j == 24):
                    sequence[j] -= 89.0
                elif (j == 4 or j == 11 or j == 18 or j == 25):
                    sequence[j] -= 100.0
                elif (j == 5 or j == 12 or j == 19 or j == 26):
                    sequence[j] -= 100.0
                elif (j == 6 or j == 13 or j == 20 or j == 27):
                    sequence[j] -= 100.0

            target = targets[i + sequenceLength - 1]
            target -= 2.0

            X.append(sequence)
            y.append(target)

        for k in range(len(xTest) - sequenceLength + 1):
            sequence = selected[k:k + sequenceLength].flatten()
            #normalizing values so they hover around 0, for some reason the switch case was giving me errors
            #so I just put a bunch of elif statements instead
            for j in range(len(sequence)):
                if (j == 0 or j % 7 == 0):
                    sequence[j] -= 29.0
                elif (j == 1 or j == 8 or j == 15 or j == 22):
                    sequence[j] -= 2.0
                elif (j == 2 or j == 9 or j == 16 or j == 23):
                    sequence[j] = (sequence[j] * 10) - 3.2
                elif (j == 3 or j == 10 or j == 17 or j == 24):
                    sequence[j] -= 89.0
                elif (j == 4 or j == 11 or j == 18 or j == 25):
                    sequence[j] -= 100.0
                elif (j == 5 or j == 12 or j == 19 or j == 26):
                    sequence[j] -= 100.0
                elif (j == 6 or j == 13 or j == 20 or j == 27):
                    sequence[j] -= 100.0

            target = targets[k + sequenceLength - 1]
            target -= 2.0

            a.append(sequence)
            b.append(target)
        return np.array(X), np.array(y), np.array(a), np.array(b)
