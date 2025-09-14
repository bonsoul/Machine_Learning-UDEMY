import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization


# load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize pixel values
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# one-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# define the baseline model
#model = Sequential([
    #Input(shape=(32, 32, 3)),
    #Conv2D(32, (3, 3), activation='relu'),
    #MaxPooling2D((2, 2)),
    #Conv2D(64, (3, 3), activation='relu'),
    #MaxPooling2D((2, 2)),
    #Flatten(),
    #Dense(128, activation='relu'),
    #Dropout(0.5),
    #Dense(10, activation='softmax')])


# compile the model
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# display the model summary
#model.summary()

# train the baseline model
#history = model.fit(
    #X_train, y_train,
    #validation_split=0.2,
    #epochs=10,
    #batch_size=64,
    #verbose=1)

#evaluate
#loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#print(f"Baseline Model Test Accuracy: {accuracy:.4f }")

#define an improved model
improved_model = Sequential([
    Input(shape=(32, 32, 3)),

    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Fully connected layers
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])


#compile the improved model with a learning rate scheduler
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
improved_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

improved_model.summary()


epochs = 10
batch_size = 64

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1.0/255.0
)



history = improved_model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch = len(X_train) // batch_size,
    validation_data=(X_test, y_test),
    epochs=epochs,
    verbose=1
)

# =========================
# Evaluate the model
# =========================
test_loss, test_acc = improved_model.evaluate(X_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# =========================
# Predictions (optional)
# =========================
y_pred = improved_model.predict(X_test[:10])
print("\nSample predictions (class probabilities for 10 test images):")
print(y_pred.argmax(axis=1))





import matplotlib.pyplot as plt

# =========================
# Plot Training History
# =========================
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
