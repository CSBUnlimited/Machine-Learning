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
dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

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




