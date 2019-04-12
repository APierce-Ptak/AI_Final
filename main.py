from mpl_toolkits.mplot3d import Axes3D
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
X, Y = np.arange(10).reshape((5, 2)), range(5)

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, Y) 
# print(neigh.predict([[0.8]]))



## Initialisation

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# df = pd.DataFrame({
#     'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
#     'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
# })


# np.random.seed(200)
# k = 3
# # centroids[i] = [x, y]
# centroids = {
#     i+1: [np.random.randint(0, 80), np.random.randint(0, 80)]
#     for i in range(k)
# }
    
# fig = plt.figure(figsize=(5, 5))
# plt.scatter(df['x'], df['y'], color='k')
# colmap = {1: 'r', 2: 'g', 3: 'b'}
# for i in centroids.keys():
#     plt.scatter(*centroids[i], color=colmap[i])
# plt.xlim(0, 80)
# plt.ylim(0, 80)
# plt.show()
