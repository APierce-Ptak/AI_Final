from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
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

# X AND Y CODE GOES HEREO
# i = 0
# outputs = []
# inputs = dataset

    # Fills output array with last column of inputs 
# for row in dataset:

#   if debug: print("BeforeRow ",i," \n",row)
#   sizeOfRow = len(row)-1
#   outputs.append(row[sizeOfRow])

#   if debug: print("\n\n\n\n\n")
  
#   i+=1
#     # input has its last column removed
# inputs = np.delete(dataset, 7, 1)

# if debug: print("Inputs:\n", inputs)
# if debug: print("\n\n\n")
# if debug: print("LastColumn:\n", outputs)

#################################

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

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, Y) 
# print(neigh.predict([[0.8]]))
print(X)


























