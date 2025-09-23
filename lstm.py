from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Dense

vocab_size = 10000
max_len = 200

(X_train, y_train),(X_test,y_test) = imdb.load_data(num_words=vocab_size)


X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_train, maxlen=max_len, padding='post')

print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: { X_test.shape}")


rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation="tanh", return_sequences=False),
    Dense(1, activation = 'sigmoid')
])

rnn_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

rnn_model.summary()


rnn_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test, y_test)


print(f"RNN Test Loss: {rnn_loss:.4f}, Test Accuracy: {rnn_accuracy:.4f}")



lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh', return_sequences=False),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

lstm_model.summary()


lstm_history = rnn_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)


lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, y_test)


print(f"LSTM Test Loss: {lstm_loss:.4f}, Test Accuracy: {lstm_accuracy:.4f}")

