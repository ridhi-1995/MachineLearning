import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv(r"C:\Users\ridhi\OneDrive\Desktop\Data sets\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values  # all the rows and columns excluding the last one
Y = dataset.iloc[:, -1].values  # only last column

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
# Polynomial Regression
poly_reg = PolynomialFeatures(degree=4)
X_ploy = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_ploy, Y)
# Show Linear Regression
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth vs Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# Show Polynomial Regression
plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg_2.predict(X_ploy), color='blue')
plt.title('Truth vs Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
# smoother curve
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth vs Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
#predecition
print(lin_reg.predict([[6.5]]))
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))