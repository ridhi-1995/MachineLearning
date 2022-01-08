import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'C:\Users\ridhi\OneDrive\Desktop\Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # all the rows and columns excluding the last one
Y = dataset.iloc[:, -1].values  # only last column
# Splitting the data set into training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# print(regressor.coef_)
y_pred = regressor.predict(X_test)
# Training set
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#plt.show()
# Test set
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
#plt.show()

# Question 1: How do I use my simple linear regression model to make a single prediction,
# for example, to predict the salary of an employee with 12 years of experience?

# Question 2: How do I get the final regression equation y = b0 + b1 x
# with the final values of the coefficients b0 and b1?

print(regressor.predict([[12]]))
print(regressor.coef_)
print(regressor.intercept_)
