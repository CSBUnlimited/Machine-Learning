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
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:, 1:2].values # Just to make x as matrix, otherwise we can use dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values # y can be a vector

# We don't have enough data to devide test data
# We don't need to miss any data cuz data is limited and very important

### In oreder to compare we create linear and polynormail regression

## Fitting Linear Regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x, y)

## Fitting Polynormial Regression
from sklearn.preprocessing import PolynomialFeatures
polynormial_features = PolynomialFeatures(degree = 2)
x_poly = polynormial_features.fit_transform(x) # first we need to fit to x and then transform to x_poly

poly_linear_regressor = LinearRegression()
poly_linear_regressor.fit(x_poly, y)

## Plotting

# Linear Regression
plt.scatter(x, y, color = 'red')
plt.plot(x, linear_regressor.predict(x), color = 'blue') # plot with x train, y predictions for x train
plt.title('Salary vs Level (Linear Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Polynormaial Regression
plt.scatter(x, y, color = 'red')
plt.plot(x, poly_linear_regressor.predict(polynormial_features.fit_transform(x)), color = 'blue') # plot with x train, y predictions for x train
plt.title('Salary vs Level (Polynormial Regression)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

## Fitting Polynormial Regression for degree 3
from sklearn.preprocessing import PolynomialFeatures
polynormial_features_3 = PolynomialFeatures(degree = 3)
x_poly_3 = polynormial_features_3.fit_transform(x) # first we need to fit to x and then transform to x_poly

poly_3_linear_regressor = LinearRegression()
poly_3_linear_regressor.fit(x_poly_3, y)

# Plotting
plt.scatter(x, y, color = 'red')
plt.plot(x, poly_3_linear_regressor.predict(polynormial_features_3.fit_transform(x)), color = 'blue') # plot with x train, y predictions for x train
plt.title('Salary vs Level (Polynormial Regression - Degree 3)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

## Fitting Polynormial Regression for degree 4
from sklearn.preprocessing import PolynomialFeatures
polynormial_features_4 = PolynomialFeatures(degree = 4)
x_poly_4 = polynormial_features_4.fit_transform(x) # first we need to fit to x and then transform to x_poly

poly_4_linear_regressor = LinearRegression()
poly_4_linear_regressor.fit(x_poly_4, y)

# Plotting
plt.scatter(x, y, color = 'red')
plt.plot(x, poly_4_linear_regressor.predict(polynormial_features_4.fit_transform(x)), color = 'blue') # plot with x train, y predictions for x train
plt.title('Salary vs Level (Polynormial Regression - Degree 4)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

# Improving plot or smothing cruves in plot
# in order to do this, currently level is increment by 1 we can reduce it to 0.1 increments
x_grid = np.arange(min(x), max(x) + 0.1, 0.1) # this will give a vector but we need is a matrix
x_grid = x_grid.reshape(len(x_grid), 1) # convert to matrix

# Plotting
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, poly_4_linear_regressor.predict(polynormial_features_4.fit_transform(x_grid)), color = 'blue') # plot with x train, y predictions for x train
plt.title('Salary vs Level (Polynormial Regression - Degree 4 | Smoothed)')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

## Predict a value using these models
predicting_value = np.array(6.5)
# reshaping to matrix 
predicting_value = predicting_value.reshape(1, 1) # convert to matrix


predications = linear_regressor.predict(predicting_value), poly_linear_regressor.predict(polynormial_features.fit_transform(predicting_value)), poly_3_linear_regressor.predict(polynormial_features_3.fit_transform(predicting_value)), poly_4_linear_regressor.predict(polynormial_features_4.fit_transform(predicting_value))










