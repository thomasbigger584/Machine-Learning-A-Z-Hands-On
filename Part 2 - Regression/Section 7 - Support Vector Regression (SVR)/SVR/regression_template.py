# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# scale matrix for features X and dependent variables y
# two separate standardscaler objects because of two separate matrices
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting the SVR Model to the dataset

from sklearn.svm import SVR

# C: t
# KERNEL: what type of svr do you want, linear, polynomial, rbf
# rbf most common, also default but add anyway

# this SVR class does not apply feature scaling
regressor = SVR(kernel = 'rbf', C = 50)
regressor.fit(X, y)

# Predicting a new result
# because we are using feature scaling, we want to scale 6.5 into something the regressor knows to predict
transformed_test_X = sc_X.transform(np.array([[6.5]]))
#this is the scalled prediction
y_pred = regressor.predict(transformed_test_X)
#we want to reverse to get the actual prediction
y_pred = sc_y.inverse_transform(y_pred)


# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('SVR. C = 50')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
