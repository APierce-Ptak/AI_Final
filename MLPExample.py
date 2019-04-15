# Example of using the scikit-learn MLP Classifier with the built-in digit
# dataself. Requires Python 3 and Scikit Learn (included with the Anaconda 3
# distribution)
#
# Used by Dr. Olsen for CS484, Spring 2017
# Modified by Dr. Simari for CS484, Spring 2019
#
# Refer to the Scikit Learn MLP Classifier Documentation:
# https://tinyurl.com/y6fx277s

from sklearn.neural_network import MLPClassifier
from sklearn import datasets
import numpy as np
import sys

import random

np.set_printoptions(threshold=sys.maxsize)

# load digit data from scikit-learn's library
# digits.images is the data, digits.target are the targets for those images
digits = datasets.load_digits()

# To apply a classifier on this data, we need to flatten the images, to turn
# the data into a (n_samples, n_features) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Split the data into training and testing
split_point = int(n_samples/2)

x_train = data[:split_point]
t_train = digits.target[:split_point]

x_test = data[split_point:]
t_test = digits.target[split_point:]

# Train the network...

print("X TRAINING DATA")
print(x_train)
print("T TRAINING DATA")
print(t_train)

# Create an MLP learner with default parameters
# TODO: modify to set optional parameters
mlp = MLPClassifier()

# Tell the learner to learn a netwok for our training data
net = mlp.fit(x_train, t_train)

# Test how well it learned...

# Get predictions; could use to build confusion matrix or other calculations
predictions = net.predict(x_test)

# Calculate accuracy
score = net.score(x_test,t_test)
print("Accuracy score: ", score)

# Number of iterations classifier ran for
print("Iterations: ", net.n_iter_)
