import numpy as np

def gradient_descent(X, y, learning_rate, num_iters):
  """
  Performs gradient descent to find optimal weights and bias for linear regression.  Args:
      X: A numpy array of shape (m, n) representing the training data features.
      y: A numpy array of shape (m,) representing the training data target values.
      learning_rate: The learning rate to control the step size during updates.
      num_iters: The number of iterations to perform gradient descent.  Returns:
      A tuple containing the learned weights and bias.
  """  # Initialize weights and bias with random values
  m, n = X.shape
  weights = np.random.rand(n)
  bias = 0  # Loop for the number of iterations
  for i in range(num_iters):
    # Predict y values using current weights and bias
    y_predicted = np.dot(X, weights) + bias    # Calculate the error
    error = y - y_predicted    # Calculate gradients for weights and bias
    weights_gradient = -2/m * np.dot(X.T, error)
    bias_gradient = -2/m * np.sum(error)    # Update weights and bias using learning rate
    weights -= learning_rate * weights_gradient
    bias -= learning_rate * bias_gradient  
    return weights, bias# Example usage
X = np.array([[1, 1], [2, 2], [3, 3]])
y = np.array([2, 4, 5])
learning_rate = 0.01
num_iters = 100
weights, bias = gradient_descent(X, y, learning_rate, num_iters)
print("Learned weights:", weights)
print("Learned bias:", bias)