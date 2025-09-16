from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


#load CIFAR 10 DATASET

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


#normalize data

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

#one hot encode the labels

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


print(f"Training Data Shape: {X_train.shape}, Labale Shape: {y_train.shape}")
print(f"Test Data Shape: {X_test.shape}, Label Shape: {y_test.shape}")


#build the CNN MODEL
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#train the model

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)


#evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#plot for loss
plt.plot(history.history['loss'], label='Training Accuracy')
plt.plot(history.history['val_loss'], label="Validation Accuracy")
plt.title("Model loss")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()