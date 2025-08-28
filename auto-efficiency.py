import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from tree.utils import *
from sklearn.tree import DecisionTreeRegressor, export_text

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

cleaned_data = data.drop('car name', axis=1)
cleaned_data.replace('?', np.nan, inplace=True)
X = cleaned_data.drop('mpg', axis = 1)
y = cleaned_data['mpg']

#3a
tree = DecisionTree(criterion="mse", max_depth=5)
tree.fit(X,y)
predicted = tree.predict(X)
print(f"RMSE from my model:",  rmse(predicted, y))

#3b
regressor = DecisionTreeRegressor(max_depth=4, )
regressor.fit(X, y)
y_pred = regressor.predict(X)
print(f"RMSE from Scikit learn:",  rmse(y_pred, y))


# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn