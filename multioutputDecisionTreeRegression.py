import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
y[::5, :] += 0.5 - rng.rand(20, 2)

regr_d2 = DecisionTreeRegressor(max_depth=2)
regr_d5 = DecisionTreeRegressor(max_depth=5)
regr_d8 = DecisionTreeRegressor(max_depth=8)

regr_d2.fit(X, y)
regr_d5.fit(X, y)
regr_d8.fit(X, y)