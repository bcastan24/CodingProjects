'''This file will be very similar to the other file in this folder but I will try to impove the model
In the older file the NN has an average difference of 0.75ish I want to get that number down
I will also be using different strategies to clean the data and be using more CS fundamentals,
this file will be less rushed and just trying to learn the fundamentals of NN, I will take my time on this one'''
import keras as keras
from keras import layers
import gatherAndCleanData as gcd

data = gcd.Data()
data.cleanData()
xTrain, yTrain = data.createTrainingSets()
print(xTrain)
