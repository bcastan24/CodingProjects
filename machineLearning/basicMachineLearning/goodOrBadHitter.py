#making a basic neural network to understand the concepts
#I did this all in another file that was in my GitHub but I moved it to this directory  and deleted the old one to group all my ML projects together
import numpy
import numpy as np

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
        #does th dot product of the array of weights and array of inputs; ((w1*x1)+(w2*x2))+b
        total = np.dot(self.weights, inputs) + self.bias
        #then put that dot product into the sigmoid activation function to get the feedforward
        return sigmoid(total)

class OurNeuralNetwork:
    #neural network with 2 inputs, 2 neurons, and 1 output layer
    def __init__(self):
        #weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        #biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        self.wh1 = np.array([self.w1, self.w2])
        self.wh2 = np.array([self.w3, self.w4])

    def feedForward(self, x):
        #where x is a numpy array with 2 elements


        h1 = sigmoid(np.dot(self.wh1, x) + self.b1)
        h2 = sigmoid(np.dot(self.wh2, x) + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, allYTrues):
        #where data is a (n x 2) numpy array and n is number of samples and where allYTrues is a numpy array with n elements
        learnRate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, yTrue in zip(data, allYTrues):
                #do a feedforward
                sum_h1 = np.dot(self.wh1, x)
                h1 = sigmoid(sum_h1)

                sum_h2 = np.dot(self.wh2, x)
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                yPred = o1

                #calculate the partial derivatives
                d_L_d_yPred = -2 * (yTrue - yPred)

                #neuron o1
                d_yPred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_yPred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_yPred_d_b3 = deriv_sigmoid(sum_o1)

                d_yPred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_yPred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                #neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                #neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                #update weights and biases
                #neuron h1
                self.w1 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w1
                self.w2 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_w2
                self.b1 -= learnRate * d_L_d_yPred * d_yPred_d_h1 * d_h1_d_b1

                #neuron h2
                self.w3 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w3
                self.w4 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_w4
                self.b2 -= learnRate * d_L_d_yPred * d_yPred_d_h2 * d_h2_d_b2

                #neuron o1
                self.w5 -= learnRate * d_L_d_yPred * d_yPred_d_w5
                self.w6 -= learnRate * d_L_d_yPred * d_yPred_d_w6
                self.b3 -= learnRate * d_L_d_yPred * d_yPred_d_b3

                #calculate total loss at end of each epoch
                if epoch % 10 == 0:
                    yPreds = np.apply_along_axis(self.feedForward, 1, data)
                    loss = mseLoss(allYTrues, yPreds)
                    #print("Epoch %d loss: %.3f" % (epoch, loss))


'''
I will be training the neural network to recognize a good baseball hitter

The data I will be feeding it is 
ops+ this season minus 100 (the average ops+ in the major leagues) 
and avgerage exit velocity - 89 (89 is the average exit velocity in the mlb) 
1 will be a good hitter and 0 will be a bad hitter
'''
data = np.array([
    [110, 6.2], #Aaron Judge
    [-53, 2.5], #Andrew Vaughn
    [95, 2.9], #Cal Raleigh
    [-46, -4.7], #Nick Madrigal 2024
    [78, 5.7], #Shohei Ohtani
    [-34, -6.4], #Jarrod Dyson 2019
])
allYTrues = np.array([
    1, #Aaron Judge
    0, #Andrew Vaughn
    1, #Cal Raleigh
    0, #Nick Madrigal 2024
    1, #Shohei Ohtani
    0, #Jarrod Dyson 2019
])

#training the neural network
network = OurNeuralNetwork()
network.train(data, allYTrues)

#make some predictions
jamesWood = np.array([66, 4.6])
michaelATaylor = np.array([-8, -1])
print("James Wood: %.3f" % network.feedForward(jamesWood))
print("Michael A Taylor: %.3f" % network.feedForward(michaelATaylor))

nameOfPlayer = input("Please input the players name: ")
opsPlusString = input("Please input their OPS+: ")
opsPlus = int(opsPlusString) - 100
exitVeloString = input("Please input their average exit velocity: ")
exitVelo = float(exitVeloString) - 89
playerArray = np.array([opsPlus, exitVelo])
print(nameOfPlayer + ": %.3f" % network.feedForward(playerArray))
if (network.feedForward(playerArray) < 0.5):
    print(nameOfPlayer + " is not a good hitter.")
else:
    print(nameOfPlayer + " is a good hitter.")




