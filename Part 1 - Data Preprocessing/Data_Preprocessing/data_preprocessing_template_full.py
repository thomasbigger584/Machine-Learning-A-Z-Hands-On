#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:42:41 2019

@author: thomasbigger
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
# colon means all of the columns, :-1 means take all but the last one
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# allows us to take care of missing data
from sklearn.preprocessing import Imputer

# also can use median or most_frequent
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# it goes [rows, columns]
# its not 1:2 but 1:3 because the upper bound is excluded. To include columns index 1 and 2, up to 3
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#however this gives numbers for each of the categories which could imply one is greater than the other
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#we need to apply it to craete a column for each category, 0 or 1 for that category
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

#for dependent data we can change the yes/nos to 0/1 similarly
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)


from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

#we need to fit then transform the training set
X_train = sc_X.fit_transform(X_train)

#we dont need to fit the test set
X_test = sc_X.transform(X_test)