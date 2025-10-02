from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN,Dense, LSTM, GRU

# load the dataset

vocab_size = 10000
max_len = 100

(X_train,y_train), (X_test, y_test) = imdb.load_data(num_words= vocab_size)



#pad sequences
X_train = pad_sequences(X_train, maxlen=max_len, padding="post")
X_test = pad_sequences(X_test, maxlen=max_len, padding="post")

print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Testing Data Shape: {X_test.shape}, {y_test.shape}")


#define the RNN Model
rnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    SimpleRNN(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])

#compile the model
rnn_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#display the model summary
rnn_model.summary()

#DEFINE THE LSTM model
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    LSTM(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])



#compile the model
lstm_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#display the model summary
lstm_model.summary()



#DEFINE THE LSTM model
gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128),
    GRU(128, activation='tanh'),
    Dense(1, activation='sigmoid')
])


#compile the model
gru_model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#display the model summary
gru_model.summary()



#train RNN Model
history_rnn =  rnn_model.fit(X_train, y_train, validation_split=0.2, epochs=5,batch_size=64, verbose=1)


#train RNN Model
history_lstm =  lstm_model.fit(X_train, y_train, validation_split=0.2, epochs=5,batch_size=64, verbose=1)


#train RNN Model
history_gru =  gru_model.fit(X_train, y_train, validation_split=0.2, epochs=5,batch_size=64, verbose=1)



#eevauate models
loss_rnn, accuracy_rnn = rnn_model.evaluate(X_test, y_test, verbose=0)

loss_lstm, accuracy_lstm = lstm_model.evaluate(X_test, y_test, verbose=0)

loss_gru, accuracy_gru = gru_model.evaluate(X_test, y_test, verbose=0)

print(f"RNN Test Accuracy: {accuracy_rnn:.4f}")
print(f"LSTM Test Accuracy: {accuracy_lstm:.4f}")
print(f"GRU Test Accuracy: {accuracy_gru:.4f}")

import matplotlib.pyplot as plt

#plot training accuracy

plt.plot(history_rnn.history['accuracy'], label="RNN Training Acuracy")
plt.plot(history_lstm.history['accuracy'], label="LSTM Training Accuracy")
plt.plot(history_gru.history['accuracy'],label="GRU Training Accuracy")

