import pandas as pd
import numpy as np
from pybaseball import batting_stats
import tensorflow as tf
import keras
from keras import layers

#Gathering and cleaning the data
#Note I copied this code from predictingStatsUsingNN.py in the basicMachineLearning folder
#defining the start and end year that we want to pull data from
START = 2004
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

batting = batting.groupby("IDfg").apply(nextSeason, include_groups=False).reset_index()

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
trainTargets = trainingData[targetColumn].values
testTargets = testingData[targetColumn].values

#create training sets
def createTrainingSets(selected, targets, sequenceLength = 4):
    X = []
    y = []
    for i in range(len(selected) - sequenceLength + 1):
        sequence = selected[i:i+sequenceLength].flatten()
        target = targets[i+sequenceLength-1]

        X.append(sequence)
        y.append(target)
    return np.array(X), np.array(y)

#create training sets
xTrain, yTrain = createTrainingSets(trainSelected, trainTargets)
xTest, yTest = createTrainingSets(testSelected, testTargets)



#make deep neural network regression
def buildAndCompileModel():
    model = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(xTrain.shape[1],)), layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'),layers.Dense(1)])
    model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001), metrics=['mae'])
    return model

nnWARModel = buildAndCompileModel()
nnWARModel.summary()

#train the model
print("\nStarting Training...")
history = nnWARModel.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=1000, batch_size=30, verbose=1)

#make predictions
predictions = nnWARModel.predict(xTest)
avgDiff = 0.0
print(f"\nSample predictions vs Actual")
for j in range(50):
    print(f"Predicted: {predictions[j][0]:.3f}, Actual: {yTest[j]:.3f}, Difference: {predictions[j][0] - yTest[j]:.3f}")
    avgDiff += predictions[j][0] - yTest[j]
print(f"Average Difference: {abs(avgDiff / 50):.3f}")