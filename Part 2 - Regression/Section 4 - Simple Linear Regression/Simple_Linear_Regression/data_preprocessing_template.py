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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling - we do not need because sklearn linear regression already has it

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# fit regressor to the training set
regressor.fit(X_train, y_train)

# predicting with the test results, the new observations
# y_test is the real salaries, y_pred is the predicted salaries
y_pred = regressor.predict(X_test)


# Visualising the training set results
# plot our observation points
plt.scatter(X_train, y_train, color='red')

# plot the line
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the test set results

# the line stays because it is a part of the regressor object

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# we can see how good the prediction is because of how many are close to the line

# for new predictions we can input new X values and get predictions for salary
