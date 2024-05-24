"""
Spencer Bledsoe

Iris Classification model using logistic regression to classify datasets into 3 separate categories.

Dataset from UCI: https://archive.ics.uci.edu/dataset/53/iris
"""



"""Importing Dependencies"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection & Processing"""

#loading the dataset to a pandas dataframe
iris = pd.read_csv('/content/bezdekIris.data', header=None)

iris.head()

iris.shape

iris.describe()

iris[4].value_counts()

iris.groupby(4).mean() # gets the mean values for all columns separated by labels. Will use this to determine the type of object detected. May need to adjust to evaluate columns separately or use different measure.

#separating data from labels (row 4)
X = iris.drop(columns=4, axis=1)
Y = iris[4]

print(X)
print(Y)

"""Training and Test Data"""

#split dataset into training and test datasets. test_size = 0.1 means we want 10% of data to be test data, stratify=Y splits data based on the number of labels, random_state=1 splits data in a specific way
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training --> Logistic Regression"""

model = LogisticRegression()

#training the Logistic Regression model using training data
model.fit(X_train, Y_train)

"""Model Evaluation"""

#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data: ', training_data_accuracy)

#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data: ', test_data_accuracy)

"""Making a Predictive System"""

#sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
input_data = (5.1, 3.3, 3.6, 1.0)

#changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if(prediction[0]=='Iris-setosa'):
  print('This is an Iris Setosa')
elif(prediction[0]=='Iris-versicolor'):
  print('This is an Iris Versicolor')
else:
  print('This is an Iris Virginica')


