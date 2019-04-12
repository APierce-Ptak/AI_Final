from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#######################
#  Nearest Neighbor   #
#  Linear Regression  #
#    Neural Net       #
#######################
debug = False

# Print Dataset files
if debug: print(os.listdir('./Dataset'))

# Input dataset
dataset = np.loadtxt(open("./Dataset/Admission_Predict_Ver1.1.csv", "rb"), delimiter=",", skiprows=1)
if debug: print(dataset)

# Clear the first column 
dataset = np.delete(dataset, 0, 1)
if debug: print(dataset)

# Print the shape of the dataset
print(dataset.shape)

# The X and Y axes of the data X what we know Y what we want to predict
X = dataset[:,0:7]
Y = dataset[:, 7]
print("X Array")
print(X)
print("Y Array")
print(Y)

# Test and training data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=None)

print("X train")
print(X_train)
print("X test")
print(X_test)
print("Y train")
print(Y_train)
print("Y test")
print(Y_test)

Y_train_reshape = Y_train.reshape(-1,1)
scalerX_train = preprocessing.StandardScaler().fit(X_train) 
scalerY_train = preprocessing.StandardScaler().fit(y_train_reshape)

Y_test_reshape = Y_test.reshape(-1,1)
scalerX_test = preprocessing.StandardScaler().fit(X_test) 
scalerY_test = preprocessing.StandardScaler().fit(y_test_reshape)

#print(scalerX_train.mean_)
#print(scalerX_train.scale_)
#print(scalerX_train.)

X, Y = np.arange(10).reshape((5, 2)), range(5)

# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(X, Y) 
# print(neigh.predict([[0.8]]))
print(X)


























