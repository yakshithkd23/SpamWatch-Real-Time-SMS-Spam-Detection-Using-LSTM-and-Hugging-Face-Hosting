import os
import gdown
import numpy as np
import re
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Constants
MODEL_URL = "https://drive.google.com/uc?export=download&id=16_6LmSP5fqTjGSjGujsre8CHR81Bhwc5"
MODEL_PATH = "lstm_model.h5"
MAX_WORDS = 5000
MAX_LEN = 100

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the model
model = load_model(MODEL_PATH)

# Initialize the tokenizer (ensure this is the same tokenizer used during training)
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")

def clean_text(text):
    """Preprocess the input text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)  # Remove numbers
    return text

def prepare_input(text):
    """Tokenize and pad the input text."""
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    return padded

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predicting spam or ham."""
    data = request.get_json()
    message = data.get("message", "")
    cleaned_message = clean_text(message)
    input_data = prepare_input(cleaned_message)
    prediction = model.predict(input_data)
    result = "spam" if prediction[0][0] > 0.5 else "ham"
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
