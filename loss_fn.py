import numpy as np

''' 
Thinking of computing the gradients in this script itself, as it requires loss function to calculate 
the grads, it might undermine the model function completely, the only work for the model is to define
the architechture of the network being used and for inference
'''

# Simple MSE using here
def compute_loss(y_pred, y): 
	n = len(y)
	loss = 1/n * ((y-y_pred)**2)
	return loss
def compute_grad(X, loss):
	n = len(X)
	dL_dw = -2/n * (np.dot(X, loss))
	dL_db = -2/n * (np.dot(1, loss))
	return dL_dw, dL_db