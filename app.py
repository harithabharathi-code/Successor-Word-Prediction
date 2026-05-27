import streamlit as st
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from doc import training_data

# Set up webpage title and text
st.title("Next Word Predictor Model")
st.write("Enter a short phrase below to let the LSTM Neural Network predict the next word.")

# Preprocess data to match the trained tokenizer settings
cleaned = re.sub(r'\W+', ' ', training_data).lower()
tokens = word_tokenize(cleaned)
train_len = 3 + 1
text_sequences = []
for i in range(train_len, len(tokens)):
    text_sequences.append(tokens[i-train_len:i])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
seq_len = 3

# Load your pre-trained model file instantly
@st.cache_resource
def load_my_lstm_model():
    return load_model("mymodel.h5")

try:
    model = load_my_lstm_model()
    
    # Create the text input box on the website
    user_input = st.text_input("Enter your phrase (3 words):", "").strip().lower()

    if user_input:
        encoded_text = tokenizer.texts_to_sequences([user_input])[0]
        if not encoded_text:
            st.warning("Words not found in the vocabulary list!")
        else:
            pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
            
            # Predict the next words
            predictions = model.predict(pad_encoded, verbose=0)[0]
            top_3_indices = predictions.argsort()[-3:][::-1]
            
            st.subheader("Top 3 Predictions:")
            for rank, index in enumerate(top_3_indices, 1):
                pred_word = tokenizer.index_word[index]
                st.write(f"**{rank}.** {pred_word}")
except Exception as e:
    st.error("Could not find 'mymodel.h5'. Please let your lstm.py training finish 50 epochs first!")
