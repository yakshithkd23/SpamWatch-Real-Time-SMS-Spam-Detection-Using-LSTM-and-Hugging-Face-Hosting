import os
import json
import re
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & tokenizer once
model = load_model(os.path.join(os.path.dirname(__file__), '../model/lstm_model.h5'))
with open(os.path.join(os.path.dirname(__file__), '../model/tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

max_len = 100

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def handler(request):
    if request.method == "POST":
        try:
            body = json.loads(request.body)
            message = body.get("message", "")
            cleaned = clean_text(message)
            sequence = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(sequence, maxlen=max_len)
            prediction = model.predict(padded)
            result = "Spam" if prediction[0][0] > 0.5 else "Ham"
            return {
                "statusCode": 200,
                "body": json.dumps({"result": result})
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": str(e)})
            }

    return {
        "statusCode": 405,
        "body": json.dumps({"error": "Method not allowed"})
    }
