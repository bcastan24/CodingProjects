#making a basic neural network to understand the concepts
import numpy as np

#making activation function
def sigmoid(x):
    #f(x)=1/(1+e^(-x))
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedForward(self, inputs):
        #does th dot product of the array of weights and array of inputs; ((w1*x1)+(w2*x2))+b
        total = np.dot(self.weights, inputs) + self.bias
        #then put that dot product into the sigmoid activation function to get the feedforward
        return sigmoid(total)