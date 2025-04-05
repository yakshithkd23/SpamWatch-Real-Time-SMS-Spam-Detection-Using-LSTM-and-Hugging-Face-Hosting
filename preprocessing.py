import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def preprocess_data(file_path, max_words=5000, max_len=100):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned_message'] = df['message'].apply(clean_text)

    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_message'])
    X = tokenizer.texts_to_sequences(df['cleaned_message'])
    X = pad_sequences(X, maxlen=max_len)
    y = np.array(df['label'])

    return X, y, tokenizer
