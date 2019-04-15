from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPRegressor #For neural net
from sklearn.linear_model import LinearRegression

import sys

#######################
#  Nearest Neighbor   #
#  Linear Regression  #
#    Neural Net       #
#######################


##########DEBUG VARS
debug = False
testMLP = False
testKNN = False
testLinReg = False

scoreMLP = -999
scoreKNN = -999
scoreLinReg = -999


#Stops truncating the data when printed
np.set_printoptions(threshold=sys.maxsize)

# Print Dataset files
if debug: print(os.listdir('./Dataset'))

# Input dataset
dataset = np.loadtxt(open("./Dataset/Admission_Predict_Ver1.1.csv", "rb"), delimiter=",", skiprows=1)
if debug: print(dataset)

# Clear the first column 
dataset = np.delete(dataset, 0, 1)
if debug: print(dataset)

# Print the shape of the dataset
if debug: print(dataset.shape)

#Left out serial number and acceptance
labels = ["GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]

# The X and Y axes of the data X what we know Y what we want to predict
X = dataset[:,0:7]
Y = dataset[:, 7]

if debug:
  print("X Array")
  print(X)
  print("Y Array")
  print(Y)

# Test and training data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)

if debug:
  print("X train")
  print(X_train)
  print("X test")
  print(X_test)
  print("Y train")
  print(Y_train)
  print("Y test")
  print(Y_test)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

if debug:
  print("AFTER SCALING X TRAINING DATA:")
  print(X_train)

  print("AFTER SCALING X TESTING DATA:")
  print(X_test)

#Y_train_reshape = Y_train.reshape(-1,1)
#scalerX_train = preprocessing.StandardScaler().fit(X_train) 
#scalerY_train = preprocessing.StandardScaler().fit(Y_train_reshape)

#Y_test_reshape = Y_test.reshape(-1,1)
#scalerX_test = preprocessing.StandardScaler().fit(X_test) 
#scalerY_test = preprocessing.StandardScaler().fit(Y_test_reshape)

#print("MEAN: ", scalerX_train.mean_)
#print("SCALE: ", scalerX_train.scale_)
#print(scalerX_train.transform(X_train))
#X_train = scalerX_train.transform(X_train)
#Y_train = scalerY_train.transform(Y_train)
#X_test = scalerX_test.transform(X_test)
#Y_test = scalerY_test.transform(Y_test)

#Y_train = Y_train.reshape(1,-1)
#Y_test = Y_test.reshape(-1,1)

#X, Y = np.arange(10).reshape((5, 2)), range(5)


#############################
#            MLP            #
#############################
if testMLP:
  mlp = MLPRegressor(hidden_layer_sizes=(500,500,500,500,500,1000))
  net = mlp.fit(X_train, Y_train)

  predictions = net.predict(X_test)

  scoreMLP = net.score(X_test, Y_test)
  print("MLP Iterations: ", net.n_iter_)


#############################
#            KNN            #
#############################
if testKNN:
  neigh = KNeighborsRegressor(n_neighbors=2)
  neigh.fit(X_train, Y_train) 

  predictionsKNN = neigh.predict(X_test)
  scoreKNN = neigh.score(X_test, Y_test)



#############################
#          LinReg           #
#############################
if testLinReg:
  linreg = LinearRegression().fit(X_train, Y_train)

  predictionsLinReg = linreg.predict(X_test)
  scoreLinReg = linreg.score(X_test, Y_test)



#############################
#       Score Summary       #
#############################
print("MLP Accuracy Score: ", scoreMLP)

print("KNN Accuracy Score: ", scoreKNN)

print("LinReg Accuracy Score: ", scoreLinReg)






