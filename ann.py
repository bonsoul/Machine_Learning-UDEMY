from tensorflow.keras.datasets import mnist, cifar10

import tensorflow as tf

import torch
import torch.nn as nn
import matplotlib.pyplot as plt



#load MNST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print(f"MNIST Dataset: Train - {X_train_mnist.shape}, Test - {X_test_mnist.shape}")

#load CIFAR10

(X_train_cifar10, y_train_cifar10), (X_test_cifar10, y_test_cifar10) = cifar10.load_data()
print(f"CIFAR1O Dataset: Train - {X_train_cifar10.shape}, Test - {X_test_cifar10.shape}")


#define a basic layer 
layer = tf.keras.layers.Dense(units=10, activation='relu')
print(f"Tensorflow : {layer}")

# pytorch

layer1 = nn.Linear(in_features=10, out_features=5)
print(f"Pytorch layer : {layer1}")

plt.imshow(X_train_mnist[0], cmap='gray')
plt.title(f"MNIST Label : {y_train_mnist[0]}")
plt.show()

#visualise CIFAR10
plt.imshow(X_train_cifar10[0], cmap='gray')
plt.title(f"CIFAR10 Label: {y_test_cifar10[0]}")
plt.show()