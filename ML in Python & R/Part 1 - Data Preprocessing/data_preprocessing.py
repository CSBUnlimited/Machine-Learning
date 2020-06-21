# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:52:00 2020

@author: chath
"""

## Data Preprocessing

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

## Takeing care of missing data
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp_mean = imp_mean.fit(x[:, 1:3])
## Replace data
x[:, 1:3] = imp_mean.transform(x[:, 1:3])

## Encoding categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Label encorder not required anymore can use ColumnTransformer
# But can use to identify the category
#labelEncoder_Country = LabelEncoder()
#x[:, 0] = labelEncoder_Country.fit_transform(x[:, 0])

# we there are multiple categpries so need to use 

columnTransformer = ColumnTransformer(
    transformers = [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],
    remainder = 'passthrough')
x = columnTransformer.fit_transform(x)

labelEncoder_Purchased = LabelEncoder()
y = labelEncoder_Purchased.fit_transform(y)

## Split dataset to train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

## Feature scaling 
# There are two ways Standardisation and Normalisation
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
# For the train data set we need to fit and transaform
x_train = sc_x.fit_transform(x_train)
# sc_x already fitted to training set, so just need transaform
x_test = sc_x.transform(x_test)

# In here we dont need feature scalizing for y cuz its a categorical varibale, will required in regression


























