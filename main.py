import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#0th layer is input layer (A0)
# Weight(matrix) dot product A0 + bias equals Z1
#Activation functions, tanh and signoid function.
#We will use the Relu. x if x > 0, 0 if x<= 0
#A1 = g(Z1) = ReLu(z1)
#Unactivated second layer Z2 = weight2 times A1 + b2 (second bias term)
#Second activation function will be the softmax activation function (e^zi)/summation(e^zj)
#Softmax activation will return a value between 0 and 1 to represent probability
#Backprop: start with predictions and find out how much deviation there is and adjust values accordingly
# dZ = A2 - Y (Prediction minus actual value)
#dW2 = 1/m dZ A1^T Derivative of loss function w respect to loss function
#db2 = the error 

data = pd.read_csv('mnist_test.csv') 
data = np.array(data)
m,n = data.shape
np.random.shuffle(data) #Randomize the data

data_dev = data[0:1000].T #Transposing the data
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T #Data that will be used in training
Y_train = data_train[0]
X_train = data_train[1:n]


def init_paramters(): #Main function to initialize all the parameters
    #Including the weights and biases
    W1 = np.random.rand(10,784) - 0.5 #First weight
    b1 = np.random.rand(10,1) - 0.5 #First bias
    W2 = np.random.rand(10,10) - 0.5 #Second weight
    b2 = np.random.rand(10,1) - 0.5 #Second bias
    return W1, b1, W2, b2

def ReLU(Z): #This is our ReLU activation function
    return np.maximum(Z, 0)

def softmax(Z): #This is our softmax activation function
    return np.exp(Z) / np.sum(np.exp(Z))

def foward_prop(W1,b1,W2,b2,X): #Foward propogation function
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y): #One hot encoding our Y matrix
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T #Take the transpose of the matrix to make it column
    return one_hot_Y

def deriv_ReLU(Z): #Taking the derivative of our RelU function
    return Z > 0 #If Z > 0, will return as 1 which is the slope of our ReLU function

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y): #Main back propogation function
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m * dZ2.dot(A1.T)
    dB2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1/m * dZ2.dot(X.T)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, dB2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2,  b2

def get_predicitons(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y): 
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iter, alpha):
    W1, b1, W2, b2 = init_paramters()
    for i in range(iter):
        Z1, A1, Z2, A2 = foward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0):
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predicitons(A2), Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)

