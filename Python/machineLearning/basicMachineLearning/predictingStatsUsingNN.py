'''
In this file I will be using the knowledge that I learned from goodOrBadHitter.py and predictingBaseballStats.py
to predict WAR of a baseball player using a simple neural network

This is not going to be an optimized neural network, so I will only train it on 10 players with 4 years of stats for
each player

I am going to feed the neural network WAR, OPS, BABIP, and wRC+ over four years and use that to make a prediction of WAR
for the next season

I am only going to use 3 neurons in the hidden layer because this is a simple NN, and with the two hidden layer neurons
and 1 output neuron I am at 50 weights

I do realize this is non-optimal, but I am still very new to NN, so I want to do a fun project to learn a bit more
'''
import pandas as pd
import numpy as np
from pybaseball import batting_stats
from pyexpat import features
from sklearn.preprocessing import MinMaxScaler

#I am going to copy a lot of code from goodOrBadHitter.py to save time, I will tweak it to fit my new project
#making activation function
def sigmoid(x):
    #f(x)=1/(1+e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    #derivative of sigmoid
    fx = sigmoid(x)
    return fx * (1-fx)

def mseLoss(yTrue, yPred):
    #mean squared error
    return ((yTrue - yPred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        #does the dot product of the array of weights and array of inputs; ((w1*x1)+(w2*x2))+b
        total = np.dot(self.weights, inputs) + self.bias
        #then put that dot product into the sigmoid activation function to get the feedforward
        return sigmoid(total)

class OurNeuralNetwork:
    #neural network with 2 inputs, 2 neurons, and 1 output layer
    def __init__(self):
        #weights
        #h1 weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        self.w11 = np.random.normal()
        self.w12 = np.random.normal()
        self.w13 = np.random.normal()
        self.w14 = np.random.normal()
        self.w15 = np.random.normal()
        self.w16 = np.random.normal()
        #h2 weights
        self.w17 = np.random.normal()
        self.w18 = np.random.normal()
        self.w19 = np.random.normal()
        self.w20 = np.random.normal()
        self.w21 = np.random.normal()
        self.w22 = np.random.normal()
        self.w23 = np.random.normal()
        self.w24 = np.random.normal()
        self.w25 = np.random.normal()
        self.w26 = np.random.normal()
        self.w27 = np.random.normal()
        self.w28 = np.random.normal()
        self.w29 = np.random.normal()
        self.w30 = np.random.normal()
        self.w31 = np.random.normal()
        self.w32 = np.random.normal()
        #h3 weights
        self.w33 = np.random.normal()
        self.w34 = np.random.normal()
        self.w35 = np.random.normal()
        self.w36 = np.random.normal()
        self.w37 = np.random.normal()
        self.w38 = np.random.normal()
        self.w39 = np.random.normal()
        self.w40 = np.random.normal()
        self.w41 = np.random.normal()
        self.w42 = np.random.normal()
        self.w43 = np.random.normal()
        self.w44 = np.random.normal()
        self.w45 = np.random.normal()
        self.w46 = np.random.normal()
        self.w47 = np.random.normal()
        self.w48 = np.random.normal()
        #o1 weights
        self.w49 = np.random.normal()
        self.w50 = np.random.normal()
        self.w51 = np.random.normal()


        #biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()

        self.wh1 = np.array([self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w7, self.w8, self.w9, self.w10, self.w11, self.w12, self.w13, self.w14, self.w15, self.w16])
        self.wh2 = np.array([self.w17, self.w18, self.w19, self.w20, self.w21, self.w22, self.w23, self.w24, self.w25, self.w26, self.w27, self.w28, self.w29, self.w30, self.w31, self.w32])
        self.wh3 = np.array([self.w33, self.w34, self.w35, self.w36, self.w37, self.w38, self.w39, self.w40, self.w41, self.w42, self.w43, self.w44, self.w45, self.w46, self.w47, self.w48])

    def feedForward(self, x):
        # where x is a numpy array with 2 elements

        h1 = sigmoid(np.dot(self.wh1, x) + self.b1)
        h2 = sigmoid(np.dot(self.wh2, x) + self.b2)
        h3 = sigmoid(np.dot(self.wh3, x) + self.b3)
        o1 = sigmoid(self.w49 * h1 + self.w50 * h2 + self.w51 * h3 + self.b4)
        return o1

    def train(self, data, allYTrues):
        # where data is a (n x 16) numpy array and n is number of samples and where allYTrues is a numpy array with n elements
        learnRate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, yTrue in zip(data, allYTrues):
                # do a feedforward
                sum_h1 = np.dot(self.wh1, x)
                h1 = sigmoid(sum_h1)

                sum_h2 = np.dot(self.wh2, x)
                h2 = sigmoid(sum_h2)

                sum_h3 = np.dot(self.wh3, x)
                h3 = sigmoid(sum_h3)

                sum_o1 = self.w49 * h1 + self.w50 * h2 + self.w51 * h3 + self.b4
                o1 = sigmoid(sum_o1)
                yPred = o1

                # calculate the partial derivatives
                d_L_d_yPred = -2 * (yTrue - yPred)

                # neuron o1
                d_yPred_d_w49 = h1 * deriv_sigmoid(sum_o1)
                d_yPred_d_w50 = h2 * deriv_sigmoid(sum_o1)
                d_yPred_d_w51 = h3 * deriv_sigmoid(sum_o1)
                d_yPred_d_b4 = deriv_sigmoid(sum_o1)

                d_yPred_d_h1 = self.w49 * deriv_sigmoid(sum_o1)
                d_yPred_d_h2 = self.w50 * deriv_sigmoid(sum_o1)
                d_yPred_d_h3 = self.w51 * deriv_sigmoid(sum_o1)

                # neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
                d_h1_d_w4 = x[3] * deriv_sigmoid(sum_h1)
                d_h1_d_w5 = x[4] * deriv_sigmoid(sum_h1)
                d_h1_d_w6 = x[5] * deriv_sigmoid(sum_h1)
                d_h1_d_w7 = x[6] * deriv_sigmoid(sum_h1)
                d_h1_d_w8 = x[7] * deriv_sigmoid(sum_h1)
                d_h1_d_w9 = x[8] * deriv_sigmoid(sum_h1)
                d_h1_d_w10 = x[9] * deriv_sigmoid(sum_h1)
                d_h1_d_w11 = x[10] * deriv_sigmoid(sum_h1)
                d_h1_d_w12 = x[11] * deriv_sigmoid(sum_h1)
                d_h1_d_w13 = x[12] * deriv_sigmoid(sum_h1)
                d_h1_d_w14 = x[13] * deriv_sigmoid(sum_h1)
                d_h1_d_w15 = x[14] * deriv_sigmoid(sum_h1)
                d_h1_d_w16 = x[15] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # neuron h2
                d_h2_d_w17 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w18 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_w19 = x[2] * deriv_sigmoid(sum_h2)
                d_h2_d_w20 = x[3] * deriv_sigmoid(sum_h2)
                d_h2_d_w21 = x[4] * deriv_sigmoid(sum_h2)
                d_h2_d_w22 = x[5] * deriv_sigmoid(sum_h2)
                d_h2_d_w23 = x[6] * deriv_sigmoid(sum_h2)
                d_h2_d_w24 = x[7] * deriv_sigmoid(sum_h2)
                d_h2_d_w25 = x[8] * deriv_sigmoid(sum_h2)
                d_h2_d_w26 = x[9] * deriv_sigmoid(sum_h2)
                d_h2_d_w27 = x[10] * deriv_sigmoid(sum_h2)
                d_h2_d_w28 = x[11] * deriv_sigmoid(sum_h2)
                d_h2_d_w29 = x[12] * deriv_sigmoid(sum_h2)
                d_h2_d_w30 = x[13] * deriv_sigmoid(sum_h2)
                d_h2_d_w31 = x[14] * deriv_sigmoid(sum_h2)
                d_h2_d_w32 = x[15] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                #neuron h3
                d_h3_d_w33 = x[0] * deriv_sigmoid(sum_h3)
                d_h3_d_w34 = x[1] * deriv_sigmoid(sum_h3)
                d_h3_d_w35 = x[2] * deriv_sigmoid(sum_h3)
                d_h3_d_w36 = x[3] * deriv_sigmoid(sum_h3)
                d_h3_d_w37 = x[4] * deriv_sigmoid(sum_h3)
                d_h3_d_w38 = x[5] * deriv_sigmoid(sum_h3)
                d_h3_d_w39 = x[6] * deriv_sigmoid(sum_h3)
                d_h3_d_w40 = x[7] * deriv_sigmoid(sum_h3)
                d_h3_d_w41 = x[8] * deriv_sigmoid(sum_h3)
                d_h3_d_w42 = x[9] * deriv_sigmoid(sum_h3)
                d_h3_d_w43 = x[10] * deriv_sigmoid(sum_h3)
                d_h3_d_w44 = x[11] * deriv_sigmoid(sum_h3)
                d_h3_d_w45 = x[12] * deriv_sigmoid(sum_h3)
                d_h3_d_w46 = x[13] * deriv_sigmoid(sum_h3)
                d_h3_d_w47 = x[14] * deriv_sigmoid(sum_h3)
                d_h3_d_w48 = x[15] * deriv_sigmoid(sum_h3)
                d_h3_d_b3 = deriv_sigmoid(sum_h3)

                # update weights and biases
                # neuron h1
                self.w1 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w1
                self.w2 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w2
                self.w3 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w3
                self.w4 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w4
                self.w5 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w5
                self.w6 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w6
                self.w7 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w7
                self.w8 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w8
                self.w9 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w9
                self.w10 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w10
                self.w11 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w11
                self.w12 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w12
                self.w13 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w13
                self.w14 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w14
                self.w15 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w15
                self.w16 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w16
                self.b1 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_b1

                # neuron h2
                self.w17 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w17
                self.w18 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w18
                self.w19 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w19
                self.w20 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w20
                self.w21 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w21
                self.w22 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w22
                self.w23 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w23
                self.w24 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w24
                self.w25 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w25
                self.w26 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w26
                self.w27 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w27
                self.w28 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w28
                self.w29 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w29
                self.w30 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w30
                self.w31 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w31
                self.w32 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w32
                self.b2 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_b2

                #neuron h3
                self.w33 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w33
                self.w34 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w34
                self.w35 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w35
                self.w36 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w36
                self.w37 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w37
                self.w38 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w38
                self.w39 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w39
                self.w40 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w40
                self.w41 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w41
                self.w42 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w42
                self.w43 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w43
                self.w44 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w44
                self.w45 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w45
                self.w46 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w46
                self.w47 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w47
                self.w48 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_w48
                self.b3 -= learnRate * d_L_d_yPred * d_yPred_d_h3 * d_h3_d_b3

                # neuron o1
                self.w49 -= learnRate * d_L_d_yPred * d_yPred_d_w49
                self.w50 -= learnRate * d_L_d_yPred * d_yPred_d_w50
                self.w51 -= learnRate * d_L_d_yPred * d_yPred_d_w51
                self.b4 -= learnRate * d_L_d_yPred * d_yPred_d_b4

                # calculate total loss at end of each epoch
                if epoch % 10 == 0:
                    yPreds = np.apply_along_axis(self.feedForward, 1, data)
                    loss = mseLoss(allYTrues, yPreds)
                    # print("Epoch %d loss: %.3f" % (epoch, loss))

#Gathering and cleaning the data
#Note I wrote this code in a test file and copied it here when I got it working
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

#scale sets
selectedScaler = MinMaxScaler()
targetScaler = MinMaxScaler()

#get selected data and convert to numpy arrays
trainSelected = trainingData[selectedColumns].values
testSelected = testingData[selectedColumns].values
trainTargets = trainingData[targetColumn].values.reshape(-1,1)
testTargets = testingData[targetColumn].values.reshape(-1,1)

#scaling selected column data
selectedTrainScaled = selectedScaler.fit_transform(trainSelected)
selectedTestScaled = selectedScaler.fit_transform(testSelected)
#scaling target column data aka Next_WAR
targetTrainScaled = targetScaler.fit_transform(trainTargets)
targetTestScaled = targetScaler.fit_transform(testTargets)

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

#create the training sets
xTrain, yTrain = createTrainingSets(selectedTrainScaled, targetTrainScaled.flatten())

#train the neural network
print("Training neural network...")
network = OurNeuralNetwork()
network.train(xTrain, yTrain)

#create predictions sets
xTest, yTest = createTrainingSets(selectedTestScaled, targetTestScaled)

if len(xTest) > 0:
    print(f"\nPredictions for all test samples:")

    allPreds = []
    allReals = []

    for j in range(len(xTest)):
        #make prediciton
        scaledPred = network.feedForward(xTest[j])
        #convert prediction to real WAR value
        realPred = targetScaler.inverse_transform([[scaledPred]])[0][0]
        actualRealWAR = targetScaler.inverse_transform([[yTest[j]]])[0][0]

        allPreds.append(realPred)
        allReals.append(actualRealWAR)

        print(f"Sample {j+1:2d}: Predicted WAR: {realPred:6.3f} | Actual WAR: {actualRealWAR:6.3f} | Difference: {abs(realPred-actualRealWAR):6.3f}")

#a function to make predictions with new data sets
def predictWAR(network, featureScaler, targetScaler, playerStats):
    playerStatsScaled = featureScaler.transform(playerStats.reshape(4, 4)).flatten()
    #make prediction
    predScaled = network.feedForward(playerStatsScaled)
    #convert back to real WAR value
    predReal = targetScaler.inverse_transform([[predScaled]])[0][0]

    return predReal


