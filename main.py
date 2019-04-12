from mpl_toolkits.mplot3d import Axes3D
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


# X, Y = np.arange(10).reshape((5, 2)), range(5)

# neigh = KNeighborsRegressor(n_neighbors=2)
# neigh.fit(X, Y) 
# print(neigh.predict([[0.8]]))
