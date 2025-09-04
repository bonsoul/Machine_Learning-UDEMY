import numpy as np


#mean squared error


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) **2)


#binary cross_entropy
def binary_cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


#example of data
y_true = np.array([1 ,0, 1, 1])
y_pred  = np.array([0.9, 0.2, 0.8, 0.7])


#CALCULATE LOSS
mse = mse_loss(y_true, y_pred)
bce = binary_cross_entropy_loss(y_true, y_pred)


print(f"MSE LOSS: {mse:.4f}")
print(f"Binary Cross-Entropy Loss: {bce:.4f}")



def mse_gradient(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


#derivative of BCE loss
def bce_gradient(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

#calculate gradients
grad_mse = mse_gradient(y_true, y_pred)
grad_bce = bce_gradient(y_true, y_pred)

print(f"MSE Gradients: {grad_mse}")
print(f"BCE Gradients: {grad_bce} ")


import matplotlib.pyplot as plt

predictions = np.linspace(0, 1, 100)
true_label = 1



#compute losses
mse_losses = [(true_label - p)**2 for p in predictions]
bce_losses = [-true_label * np.log(max(p, 1e-15)) - (1 - true_label) * np.log(max(1-p, 1e-15)) for p in predictions]



#plot
plt.figure(figsize=(10,6))
plt.plot(predictions, mse_losses, label="MSE Loss")
plt.plot(predictions, bce_losses, label="Binary Cross Entropy Loss")
plt.title('Loss Function Comparisons')
plt.xlabel('Prediction')
plt.ylabel('loss')
plt.legend()
plt.show()