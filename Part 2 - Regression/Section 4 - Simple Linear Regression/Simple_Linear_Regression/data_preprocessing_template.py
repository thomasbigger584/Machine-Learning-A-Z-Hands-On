# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling - we do not need because sklearn linear regression already has it

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# fit regressor to the training set
regressor.fit(X_train, y_train)

# predicting with the test results
y_pred = regressor.predict(X_test)


# Visualising the training set results
plt.scatter(X_train, y_train, color='red')

