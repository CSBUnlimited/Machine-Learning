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
dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

## Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

columnTransformer = ColumnTransformer(
    transformers = [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder = 'passthrough')
x = columnTransformer.fit_transform(x)

## Avoiding the dummy variable trap
x = x[:, 1:] # we remove one of categorical dummy variable to prevent dummy variable trap

## Split dataset to train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

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

## Fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

## Predicting the test set results
y_pred = regressor.predict(x_test)

## Building the optimal model using Backward elimination
import statsmodels.api as sm
# Add coumns of ones
x = np.append(arr = np.ones((50, 1)).astype(int), values = x, axis = 1)
# Create optimal matris of features- only contain highly impacting variables
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_opt = np.array(x_opt, dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# Let's take significate level as 0.05 which is 5%
# In backward elimination remove first hight P valued parameter
# x2 has 0.990 which is 99%, which greater than 5% so it need to remove
x_opt = x[:, [0, 1, 3, 4, 5]]
x_opt = np.array(x_opt, dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# x1 has 0.940 which is 94%, which greater than 5% so it need to remove
x_opt = x[:, [0, 3, 4, 5]]
x_opt = np.array(x_opt, dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# x2 has 0.602 which is 60.2%, which greater than 5% so it need to remove
x_opt = x[:, [0, 3, 5]]
x_opt = np.array(x_opt, dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# x2 has 0.060 which is 6%, which greater than 5% so it need to remove
x_opt = x[:, [0, 3]]
x_opt = np.array(x_opt, dtype = float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()























