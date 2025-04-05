import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess data
file_path = "sma.csv"
max_words = 5000
max_len = 100
X, y, tokenizer = preprocess_data(file_path, max_words, max_len)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save tokenizer and model
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

model.save('lstm_model.h5')

# Save training history for plotting
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Save test set for evaluation
np.savez('test_data.npz', X_test=X_test, y_test=y_test)
