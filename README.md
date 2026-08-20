# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the California Housing dataset by selecting input features and two output variables, then split the data into training and testing sets.

2. Scale the training and testing data using StandardScaler to bring the features and target values to a common scale.

3. Train the models using MultiOutputRegressor with SGDRegressor, and also train separate LinearRegression and SGDRegressor models.

4. Predict and evaluate the results by generating predictions on the test data and converting the scaled predictions back to the original scale.
 

## Program:
```
Developed by: SRIRAM ARUN S
Register No : 212225040429
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()

X = housing.data[:, :3]

Y = np.column_stack((housing.target, housing.data[:, 6]))

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)


sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model = MultiOutputRegressor(sgd)

model.fit(X_train_scaled, Y_train_scaled)

Y_pred_scaled = model.predict(X_test_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)


print("Actual Values:")
print(Y_test[:10]) 

print("\nPredicted Values:")
print(Y_pred[:10]) 

y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)
```

## Output:
<img width="513" height="631" alt="image" src="https://github.com/user-attachments/assets/64cbcdf0-f7c4-48cc-ad66-4cb4c75b9501" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
