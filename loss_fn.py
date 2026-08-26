import numpy as np

''' 
Thinking of computing the gradients in this script itself, as it requires loss function to calculate 
the grads, it might undermine the model function completely, the only work for the model is to define
the architechture of the network being used and for inference
'''

# Simple MSE using here
def compute_loss(y_pred, y): 
	n = len(y)
	# loss = 1/n * ((y-y_pred)**2)
	# We will ignore calculating MSE directly, as we are using thast info in grad calc.
	# So i will simply calc. y-y_pred as i dont directly need mse anywhere except for 
	# maybe printing the value the mse loss
	loss = (y-y_pred)
	return loss
def compute_grad(X, loss):
	n = len(X)
	dL_dw = -2/n * (X.T @ loss)
	dL_db = -2/n * (np.sum(loss))
	return dL_dw, dL_db