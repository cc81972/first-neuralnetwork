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
print(data.head())
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
