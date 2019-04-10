from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

debug = True

# Print Dataset files
if debug: print(os.listdir('./Dataset'))

x = np.loadtxt(open("./Dataset/Admission_Predict_Ver1.1.csv", "rb"), delimiter=",", skiprows=1)
print(x)

# BEHOLD THE OLD CODE
# df1 = pd.read_csv('./Dataset/Admission_Predict.csv', delimiter=',')
# df1.dataframeName = 'Admission_Predict.csv'

# More data so we'll use this one
# df1 = pd.read_csv('./Dataset/Admission_Predict_Ver1.1.csv', delimiter=',')
# df1.dataframeName = 'Admission_Predict_Ver1.1.csv'

# Drop Serial No. Column as datafrom include index
# df1.drop(df1.columns[0],axis=1,inplace=True)
# print(df1.head(5))

# nRow, nCol = df1.shape
# print("There are "+ str(nRow) + " rows and " + str(nCol) + " columns" )

# Correlation matrix
# def plotCorrelationMatrix(df, graphWidth):
#     filename = df.dataframeName
#     df = df.dropna('columns') # drop columns with NaN
#     df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
#     if df.shape[1] < 2:
#         print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
#         return
#     corr = df.corr()
#     plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
#     corrMat = plt.matshow(corr, fignum = 1)
#     plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
#     plt.yticks(range(len(corr.columns)), corr.columns)
#     plt.gca().xaxis.tick_bottom()
#     plt.colorbar(corrMat)
#     plt.title(f'Correlation Matrix for {filename}', fontsize=15)
#     plt.show()

# plotCorrelationMatrix(df1, 10)

# print(df1.loc[:, "University Rating"])
# print(df1.loc[:, "Chance of Admit "])
