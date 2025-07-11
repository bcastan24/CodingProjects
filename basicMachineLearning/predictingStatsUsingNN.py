'''
In this file I will be using the knowledge that I learned from goodOrBadHitter.py and predictingBaseballStats.py
to predict WAR of a baseball player using a simple neural network

This is not going to be an optimized neural network, so I will only train it on 30 players with 4 years of stats for
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