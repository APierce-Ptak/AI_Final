from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPRegressor #For neural net
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets

import sys

#######################
#  Nearest Neighbor   #
#  Linear Regression  #
#    Neural Net       #
#######################


##########DEBUG VARS
debug = False

# If these are false they output fake data
testMLP = True
testKNN = True
testLinReg = True

# Fake data
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

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

#############################
#            MLP            #
#############################
if testMLP:
  mlp = MLPRegressor(hidden_layer_sizes=(46,54,105),activation="identity", solver="sgd",learning_rate="adaptive",alpha=0.9999999,early_stopping=False,max_iter = 500,random_state=42)
  net = mlp.fit(X_train, Y_train)

  predictions = net.predict(X_test)

  scoreMLP = net.score(X_test, Y_test)
  print("MLP Iterations: ", net.n_iter_)



############################# 
#            KNN            # 
############################# 
if testKNN:
  neigh = KNeighborsRegressor(n_neighbors=8)
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
print("MLP Accuracy Score:\t", scoreMLP)

print("KNN Accuracy Score:\t", scoreKNN)

print("LinReg Accuracy Score:\t", scoreLinReg)



#############################
#        GridSelect         #
#############################
MLP_Parameters = {
  'hidden_layer_sizes': [(30,30,30),(50,50,50),(70,70,70),(90,90,90),(150,150,150),(250,250,250),(50,50,50,50)],
  'activation': ['relu','tanh','logistic','identity'],
  'solver': ['adam','sgd','lbfgs'],
  'alpha': [0.0001, 0.0005, 0.001, 0.01, 0.05, 0.1],
  'learning_rate': ['constant', 'invscaling', 'adaptive'],
  'epsilon': [0.00000001, 0.0000001, 0.000001, 0.00001],
}

KNN_Parameters = {
  'n_neighbors': [1,2,3,4,5,6,7,8,9,10],
  'weights': ['uniform', 'distance'],
  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
  'leaf_size': [15, 30, 50, 100],
  'p': [1,1.5,2],
}

LinReg_Parameters = {
  "NONE"
}

#svc = svm.SVC(gamma="scale")
#mlp_clf = GridSearchCV(MLP_Parameters, cv=5)
#mlp_clf.fit(X_train, Y_train)
#print("BEST PARAMS FOR MLP:")
#print(mlp_clf.best_params_)
#print("BEST SCORE FOR MLP:")
#print(mlp_clf.best_score_)

#knn_clf = GridSearchCV(KNN_Parameters, cv=5)
#knn_clf.fit(X_train, Y_train)
#print("BEST PARAMS FOR KNN:")
#print(knn_clf.best_params_)
#print("BEST SCORE FOR KNN:")
#print(knn_clf.best_score_)














