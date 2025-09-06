# importance of learning rate and choosing the right optimizer

#Learning Rate:
# Dtermines the step size for parameter updates.
#Too high :May overshoot the minimum or cause divergence
#Too Low : Leads to slow convergence

# Choosing the Right Otimizer
#  SGD: Works well for simple, convex problems
# Adam:Generally performs well across tasks.
#RMSprop: Often preferred for RNNs and sequence-based tasks.


import numpy as np
import matplotlib.pyplot as plt

#generate data

np.random.seed(42)
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.randn(100, 1)

#visualize the data


#plt.scatter(X,y, color="blue")
#plt.title('Generated Dataset')
#plt.xlabel('X')
#plt.ylabel('y')
#plt.grid()
#plt.show()


#parameters
m = 100
theta = np.random.randn(2,1)
learning_rate = 0.1
iterations = 1000


# add bias term to x
X_b = np.c_[np.ones((m, 1)), X]



#gradient
for iteration in range(iterations):
    gradients = 2/m *X_b.T.dot(X_b.dot(theta) - y)
    theta -= learning_rate * gradients
    
print("Optimized Parameters (Theta): \n", theta)




import tensorflow as tf

# prepare the data
X_tensor = tf.constant(X, dtype=tf.float32)
y_tensor = tf.constant(y, dtype=tf.float32)

# define the model
class LinearModel(tf.Module):
    def __init__(self):
        self.weights = tf.Variable(tf.random.normal([1, 1]))  # shape (1,1)
        self.bias = tf.Variable(tf.random.normal([1]))
        
    def __call__(self, X):
        return tf.matmul(X, self.weights) + self.bias
    
# define loss function
def mse_loss(y_true, y_predict):
    return tf.reduce_mean(tf.square(y_true - y_predict))

# train with SGD
model = LinearModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

#for epoch in range(100):
    #with tf.GradientTape() as tape:
        #y_pred = model(X_tensor)
        #loss = mse_loss(y_tensor, y_pred)
    #gradients = tape.gradient(loss, [model.weights, model.bias])
    #optimizer.apply_gradients(zip(gradients, [model.weights, model.bias]))
    #if epoch % 10 == 0:
        #print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}")



import torch
import torch.nn as nn
import torch.optim as optim


X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)


#define model

class LineaModelTorch(nn.Module):
    def __init__(self):
        super(LineaModelTorch, self).__init__()
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)
    
model_torch = LineaModelTorch()


#define los function


criterion = nn.MSELoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.1)


#train mdel
for epoch in range(0,100):
    optimizer.zero_grad()
    outputs = model_torch(X_torch)
    loss = criterion(outputs, y_torch)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss: {loss.item():.4f}")