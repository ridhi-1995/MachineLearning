import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv(r"C:\Users\ridhi\OneDrive\Desktop\Data sets\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values  # all the rows and columns excluding the last one
Y = dataset.iloc[:, -1].values  # only last column
print(X)
print(Y)
print(X.ndim)
print(Y.ndim)
# Reshape Y in 2D array
Y = Y.reshape(len(Y), 1)
print(Y)
print(Y.ndim)
#Feature scalling
sc_X = StandardScaler()
sc_Y = StandardScaler()
X_ft = sc_X.fit_transform(X)
Y_ft = sc_Y.fit_transform(Y)
print(X_ft)
print(Y_ft)
print(X_ft.ndim)
print(Y_ft.ndim)

#Apply SVR model
regressor = SVR(kernel = 'rbf')
regressor.fit(X_ft, Y_ft)
sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#plot
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict((X_grid)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()