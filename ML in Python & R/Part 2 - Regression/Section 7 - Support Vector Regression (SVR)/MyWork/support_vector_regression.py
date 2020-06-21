# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:57:00 2020

@author: chath
"""

## Data Preprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values # Just to make x as matrix, otherwise we can use dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values # y can be a vector


## Split dataset to train and test
"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
"""


## Feature scaling 
# There are two ways Standardisation and Normalisation
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(len(y), 1))

## Fitting Regression Model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # Gausian kernal
regressor.fit(x, y)

## Predict
# predicting_value = np.array(6.5)
# predicting_value = predicting_value.reshape(1, 1) # convert to matrix
# Can perform above in one line
predicting_value = np.array([[6.5]])
# We performed feature scaling so we need to perfom tat here too
predicting_value = sc_x.transform(predicting_value)

y_pred = regressor.predict(predicting_value)
y_pred = sc_y.inverse_transform(y_pred)

## Visualization
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue') # plot with x train, y predictions for x train
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()


## Visualization with higher smooth
x_grid = np.arange(min(x), max(x) + 0.1, 0.1) # this will give a vector but we need is a matrix
x_grid = x_grid.reshape(len(x_grid), 1) # convert to matrix

plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue') # plot with x train, y predictions for x train
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()



