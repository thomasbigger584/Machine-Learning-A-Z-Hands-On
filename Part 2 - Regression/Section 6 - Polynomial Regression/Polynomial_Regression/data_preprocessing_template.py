# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# we dont actually need the first column because we have the level column, they are basically the same
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# X is a vector (array) but we want them considered as a matrix always for the model
# this is why above we use 1:2, so it takes only that column, the 2 is the upper bound so only takes a single column
# y is ok to be a vector array

# we want to predictthe salray for 6.5 level (x variable)


# We dont want to split the train and test because we need all the values to make an accurate prediction, 
# we dont want to miss the target. we need the maximum amount of information

# same library to use to linear regression
# we will compare linear and polynomial regression to see which is more accurate. polynomial should be
from sklearn.linear_model import LinearRegression
# linear regressor
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures

# this will transform X into a new matrix, to all the different powers up to n
# i.e. X, X^2, X^3...X^n
poly_reg = PolynomialFeatures(degree = 4)
#create the new matrix from X, by fitting x then transform
X_poly = poly_reg.fit_transform(X)
# notice the constant b0 is automatically added from the formula, the column of 1's as the first column
# notice X is now the second column, and the second column is x^2

# fit the new X_poly in instead of X
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# Comparing Linear and Polinomial Regression

# Visualising the Linear Regression Results

# plot all the observation points (the true results)
plt.scatter(X, y, color = 'red')

x_lin_pred = lin_reg.predict(X)
# plot the prediction
plt.plot(X, x_lin_pred, color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# this is a straight line, because it is a linear regression model
# it does not fit the curve, so we will get wrong results
# for example position level 10 is not accurate, should be near 100000 and not around 60000


# Visualising the Polynomial Regression Results

# Dont use the existing X_poly becaue it was already defined for an exising matrix of features X
# We want to make this code applicable to any new matrix of features X so call the fit transform with X again
# This generalises things and makes it more useful going forward

plt.scatter(X, y, color = 'red')
x_lin_pred_2 = lin_reg_2.predict(poly_reg.fit_transform(X))
plt.plot(X, x_lin_pred_2, color='blue')
plt.title('Polynomial Regression. Degree = 4')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Now we have a curve, so this is not linear (not a straight line)
# we can see that we can get more accurate results for 

# we want to do better, so we can change the degree, changing from 2, to 3 and get a better fit
# however degree = 4 provides the best fit
