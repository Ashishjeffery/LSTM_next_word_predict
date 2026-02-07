import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 20  # must match training

# Load the LSTM Model
model = load_model('next_word_lstm.keras', compile=False)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text):
    token_list = tokenizer.texts_to_sequences([text])[0]

    token_list = token_list[-SEQUENCE_LENGTH:]

    token_list = pad_sequences(
        [token_list],
        maxlen=SEQUENCE_LENGTH,
        padding='pre'
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word

    return None

# Streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text = st.text_input("Enter the sequence of Words", "To be or not to")

if st.button("Predict Next Word"):
    next_word = predict_next_word(model, tokenizer, input_text)
    st.write(f"Next word: {next_word}")
