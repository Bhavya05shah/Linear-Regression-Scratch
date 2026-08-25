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
	