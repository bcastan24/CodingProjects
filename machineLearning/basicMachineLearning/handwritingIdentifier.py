import tensorflow as tf

import keras
import numpy as np

#get data in this case it is from keras
mnist = keras.datasets.mnist
#the mnist data from keras loads into two tuples
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

#normalize the data right now the data has values between 0 and 255 but we want to normalize it so the values are
#between 0 and 1
xTrain, xTest = xTrain / 255, xTest / 255

#make model
model = keras.models.Sequential([
    #helps determine the input size
    keras.layers.Flatten(input_shape=(28,28)),
    #128 neurons in the hidden layer
    keras.layers.Dense(128, activation='relu'),
    #10 output neurons because we are getting a number 0-9
    keras.layers.Dense(10)
])

model.summary()