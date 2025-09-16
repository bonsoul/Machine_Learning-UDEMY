import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


#load the sample grayscale image

image = np.random.rand(10,10)


print(image)

#define cnn kernels
edge_detection_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

blur_kernel = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
]) / 9

edge_detected_image = convolve(image, edge_detection_kernel)
blurred_image = convolve(image, blur_kernel)


#visualize original

#fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#axes[0].imshow(image, cmap="gray")
#axes[0].set_title("Original Image")
#axes[1].imshow(edge_detected_image, cmap="gray")
#axes[1].set_title("Edge Detected")
##axes[2].imshow(blurred_image, cmap="gray")
#axes[2].set_title("Blurred")
#plt.show()



import tensorflow as tf

image_tensor = tf.random.normal([1,10,10,1])


#define a convolutional layer
conv_layer = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3,3),
    activation="relu",
    input_shape=(28,28,1)
)



#applying convolution
#output_tensor = conv_layer(image_tensor)

#print(f"Original Shape: {image_tensor.shape}")
#print(f"Output Shape: {output_tensor.shape}")




#pytorch

import torch
import torch.nn as nn

#create a smaple input tensor

image_torch = torch.randn(1, 1, 10, 10)

#define a convolution layer
conv_torch = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=1
)


output_tensor_pt = conv_torch(image_torch)

print(f"Original Shape: {image_torch.shape}")
print(f"Output Shape: {output_tensor_pt}")