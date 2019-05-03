
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

# we arent using index 0 or 1 for this so only take indexes 2 and 3
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

# we want 300 in training and 100 in test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# we need to apply feature scaling because we want accurate predictions
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting our training data into the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the Test set results
#vector of each of the test set observations
y_pred = classifier.predict(X_test)

# Making the confusion matrix
# evaluate the accuracy of a classifcation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# we can see, the correct and incorrect predictions
# 65 and 24 are correct, 8 and 3 are incorrect

# Visualising the training set results

# Red and Green Points. Observation points of our trainging set
# Red who is 0 and Green who is 1
# We can see that the users who are young and low estimated salary didnt take the action
# The users who are older and higher salary took the action
# we are trying to make a classifier by placing a new observation into the prediction region
# separated by a straight line to separate the prediction regions
# if its a straight line then its a LINEAR CLASSIFIER
# we will see that non linear classifiers will not have a straight line
# however it has trouble catching points where green is in the red region and vice versa
# those incorrect predictions because we have a linear classifier and our data is not linearly distrubuted
#in order to get an absolute correct classification the line would need to be curved
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()