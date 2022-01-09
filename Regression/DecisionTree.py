import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv(r"C:\Users\ridhi\OneDrive\Desktop\Data sets\Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values  # all the rows and columns excluding the last one
Y = dataset.iloc[:, -1].values  # only last column

regressor = DecisionTreeRegressor(random_state=0)
regressor = regressor.fit(X, Y)
print(regressor.predict([[6.5]]))

#plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth vs Bluff(Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
