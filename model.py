import numpy as np
import random as random

class LinearRegression():
	def __init__(self, n_features):
		self.w = np.zeros((n_features, 1))
		self.b = 0.0
	def set_params(self, new_w, new_b):
		self.w = new_w
		self.b = new_b
	def predict(self, X):
		return np.dot(X, self.w) + self.b

class LinearRegressionClosedForm():
	def __init__(self):
		self.w = None
		self.b = 0.0
	def fit(self, X, y):
		X = np.array(X)
		y = np.array(y)

		# Augmented X has the weights and bias as theta, treating the bias as a new column
		X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
		A = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y
		self.b = A[0]
		self.w = A[1:]
	def predict(self, X):
		X = np.array(X)
		return  X @ self.w + self.b
	