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
        for i in range(len(selected) - sequenceLength + 1):
            sequence = []
            for j in range(sequenceLength):
                temp = selected[i:i+j]
                temp.col["Age"] -= 29.0
                temp.col["WAR"] -= 2.0
                temp.col["wOBA"] = (temp.col["wOBA"] * 10.0) - 3.2
                temp.col["EV"] -= 89.0
                temp.col["BABIP+"] -= 100
                temp.col["OBP+"] -= 100
                temp.col["wRC+"] -= 100
                temp.append(sequence)
            target = targets[i + sequenceLength - 1]

            X.append(sequence)
            y.append(target)
        return np.array(X), np.array(y)
