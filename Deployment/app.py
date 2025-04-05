import gradio as gr
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔵 Load trained LSTM model and tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 100

# 🔵 Prediction logic
def predict_spam(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0][0]
    return "🚫 <b>Spam</b>" if pred > 0.5 else "✅ <b>Not Spam</b>"

# 🔵 Custom CSS for styling
custom_css = """
h1 {
  color: #111827;
  font-size: 2rem;
  font-weight: bold;
  margin-bottom: 1rem;
  text-align: center;
}
body {
    font-family: 'Poppins', sans-serif;
    background-color: #f0f4f8;
    color: #1f2937;
}
textarea, input[type="text"] {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
    font-family: 'Poppins', sans-serif;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}
button {
    background-color: #4f46e5;
    color: white;
    padding: 12px 24px;
    border: none;
    border-radius: 10px;
    font-family: 'Poppins', sans-serif;
    font-size: 16px;
    cursor: pointer;
    transition: 0.2s ease;
}
button:hover {
    background-color: #4338ca;
}
.output-box {
    font-size: 22px;
    font-weight: 600;
    padding: 14px;
    border-radius: 12px;
    background-color: #e0f2fe;
    color: #dc2626; /* 🔴 Red text for visibility */
    margin-top: 16px;
    text-align: center;
    border: 2px solid #93c5fd;
}
"""

# 🔵 Gradio App Layout
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
    <h1>📲 Spam SMS Detection</h1>
    <p style="font-size: 18px; color: #1f2937; text-align:center;">
        AI-powered app to detect if an SMS message is spam or safe
    </p>
    """)

    with gr.Column():
        message = gr.Textbox(
            placeholder="✉️ Type your SMS message here...",
            label="Enter SMS Message",
            lines=3
        )
        button = gr.Button("🔍 Check for Spam")
        result = gr.HTML("")

        def display_prediction(text):
            prediction = predict_spam(text)
            return f"<div class='output-box'>{prediction}</div>"

        button.click(display_prediction, inputs=message, outputs=result)

demo.launch()
