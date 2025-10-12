import tensorflow as tf
from tensorflow.keras.applications import ResNet50

#LOAD THE MODEL
model = ResNet50(weights="imagenet")

#dispaly
model.summary()