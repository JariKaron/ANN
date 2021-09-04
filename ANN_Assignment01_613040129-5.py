"""
Assignment 1: custom-made model on data of choice
Student name: Thiranat Kanchanophat
ID : 613040129-5
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from openpyxl import workbook

"""Local of the data"""
raw_dataset = pd.read_excel("C:/Users/Admin/Documents/raw_dataset.xlsx");
print(raw_dataset)

"""Check table of the data when pandas local of the data"""
raw_dataset.head(5)
raw_dataset.tail(5)
raw_dataset.describe()

"""Train & Test Set"""
x = raw_dataset["Weight"].values.reshape(-1, 1)
y = raw_dataset["MPG"].values.reshape(-1, 1)

"""The variance of MPG is between 80% - 20%"""
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=0)

"""Training"""
model = LinearRegression()
model.fit(x_train, y_train)

"""Test"""
# y_pred is the predicted result from the model.
# y_test is the actual data used to test the model prediction.
y_pred = model.predict(x_test)

"""Hyperparameters"""
plt.scatter(x_test, y_test)
plt.plot(x_test, y_pred, 'r')


"""Compare true data & predict data"""
# When we want to make the chart to compare true data & predict data
df = pd.DataFrame({'Actually': y_test.flatten(), 'Predicted': y_pred.flatten()})
df1 = df.head(20)
df1.plot(kind="bar", figsize=(16, 10))
plt.show()

"""Total Compare true data & predict data"""
# It will show when close the tab of hyperparameters and the chart
# The process will finished
print("Mean Absolute Error(MAE) = ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error(MSE) = ", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error(RMSE) = ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Score = ",metrics.r2_score(y_test, y_pred))
