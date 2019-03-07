# Data Preprocessing Template


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#however this gives numbers for each of the categories which could imply one is greater than the other
#change text into numbers
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

#we need to apply it to craete a column for each category, 0 or 1 for that category
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the dummy variable trap
#remove the first column as it is one of the categories, we want to remove one of the categories  columns as its inferred
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set Results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# add a column of 1's at the start so as it goes with the linear regression formula b0 constant
# this is done in preparation to backwards elimination
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# will contain only variables which have high significance later
X_opt = X[:, [0, 1, 2, 3, 4, 5]]

# STEP 2: fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

# STEP 3: consider the predictor with the highest p-value and remove if P > SL
# contains useful metrics to make model more robust. has the P values.
# the lower the p value the more significant the independant variable is going to be WRT the dependant variables
# const = x0 = 1 we added previous
# x1, x2 = 2 categorical dummy variables for state
# x3 = R&D Spend
# x4 = Admin Spend
# x5 = Marketing Spend

# look for the highest p value, it is x2 with 0.990, most above SL = 0.05, we need to remove x2
print(regressor_OLS.summary())


# STEP 4 removing index 2, as it is x2 with the highest p value
X_opt = X[:, [0, 1, 3, 4, 5]]

# STEP 5: fit the full model with all remaining predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

# Iterate this until all P < SL

# ----------------------------------------------

# STEP 4 removing index 1, x1
X_opt = X[:, [0, 3, 4, 5]]

# STEP 5: fit the full model with all remaining predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())


# In the case of P alue = 0.000, the P value is so small it is rounded to 0.00

# STEP 4 removing index 2, x2
X_opt = X[:, [0, 3, 5]]

# STEP 5: fit the full model with all remaining predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())


# STEP 4 removing index 2, x2
X_opt = X[:, [0, 3]]

# STEP 5: fit the full model with all remaining predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())