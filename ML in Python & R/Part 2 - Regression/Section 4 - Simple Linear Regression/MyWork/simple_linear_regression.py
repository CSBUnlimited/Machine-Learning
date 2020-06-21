# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:07:22 2020

@author: chath
"""

## Data Preprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset
dataset = pd.read_csv('Salary_Data.csv')

x = dataset.iloc[:, :-1].values # only remove last column
y = dataset.iloc[:, 1].values

## Split dataset to train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

## Feature scaling 
"""
# There are two ways Standardisation and Normalisation
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
# For the train data set we need to fit and transaform
x_train = sc_x.fit_transform(x_train)
# sc_x already fitted to training set, so just need transaform
x_test = sc_x.transform(x_test)
"""

## Simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

## Predict the Test set results
y_pred = regressor.predict(x_test)

## Visulalizing the trainging set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # plot with x train, y predictions for x train
plt.title('Salary vs Experience (Traning set)')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show()

## Visulalizing the test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # We don't need to change this test data cuz it's already trained using train data
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.show()

