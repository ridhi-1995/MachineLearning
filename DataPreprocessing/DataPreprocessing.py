import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv(r'C:\Users\ridhi\Downloads\Machine+Learning+A-Z+(Codes+and+Datasets)\Machine Learning A-Z ('
                      r'Codes and Datasets)\Part 1 - Data Preprocessing\Section 2 -------------------- Part 1 - Data '
                      r'Preprocessing --------------------\Python\Data.csv')
X = dataset.iloc[:, :-1].values  # all the rows and columns excluding the last one
Y = dataset.iloc[:, -1].values  # only last column
print('Independent Variables')
print(X)
print('Dependent Variables')
print(Y)
print('--------------------------------------------')
print()

# Handling missing values
print('Handling missing values')
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
print('--------------------------------------------')
print()


# Encoding Categorical data, one hot coding
# Encoding independent variable
print('Encoding independent variable')
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
print('--------------------------------------------')
print()

# Encoding dependent variable
print('Encoding dependent variable')
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
print('--------------------------------------------')
print()

# Splitting the data set into training set and test set
# apply feature scaling after splitting , test set is suppose to be brand
# new set, feature scaling is a technique that will get mean SD of the feature

print('Splitting the data set into training set and test set')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print('Training set')
print('Independent Variables')
print(X_train)
print('Dependent Variables')
print(Y_train)
print()
print('Test set')
print('Independent Variables')
print('Dependent Variables')
print(X_test)
print(Y_test)
print('--------------------------------------------')
print()

# feature scaling
print('feature scaling')
# do we have to apply feature scaling to dummy variables in the matrix of features
# No, standardization to have in same range, -3 to +3
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print('Training set')
print(X_train)
print('Test set')
print(X_test)


regressor = LinearRegression()
regressor.fit(X_train, Y_train)