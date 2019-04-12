from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

debug = True

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
if debug: print(X)
if debug: print(Y)

# Test and training data set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=None)


<<<<<<< HEAD
X = dataset[:,0:7]
Y = dataset[:, 7]
print(X)
print(Y)

Yreshape = Y.reshape(-1,1)
scalerY = preprocessing.StandardScaler().fit(Yreshape) 
scalerX = preprocessing.StandardScaler().fit(X)
print("Scalar is: ", scalerX)
print("Scalar is: ", scalerY)

X, Y = np.arange(10).reshape((5, 2)), range(5)
=======
# X, Y = np.arange(10).reshape((5, 2)), range(5)
>>>>>>> fd6fca7c9ccde30dfcc097dc1d6726dadea21a6f

# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(X, Y) 
# print(neigh.predict([[0.8]]))
print(X)


























