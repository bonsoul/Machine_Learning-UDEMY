from tensorflow.keras.datasets import mnist, cifar10

import tensorflow as tf



#load MNST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
print(f"MNIST Dataset: Train - {X_train_mnist.shape}, Test - {X_test_mnist.shape}")

#load CIFAR10

(X_train_cifar10, y_train_cifar10), (X_test_cifar10, y_test_cifar10) = cifar10.load_data()
print(f"CIFAR1O Dataset: Train - {X_train_cifar10.shape}, Test - {X_test_cifar10.shape}")


#define a basic layer 
layer = tf.keras.layers.Dense(units=10, activation='relu')
print(f"Tensorflow : {layer}")