# 🔮 Next Word Prediction using LSTM (Shakespeare Text)

This project implements a **Next Word Prediction system** using an **LSTM-based deep learning model** trained on Shakespeare’s *Hamlet*.  
The model predicts the most probable next word given a sequence of words.

A **Streamlit web application** is included for interactive testing.

---

## 📌 Project Overview

- **Task**: Next Word Prediction (Language Modeling)
- **Model**: LSTM with Embedding Layer
- **Dataset**: Shakespeare – *Hamlet*
- **Frameworks**: TensorFlow / Keras
- **Frontend**: Streamlit

---

## 🧠 Key Concepts Used

- Tokenization with vocabulary limiting
- Trainable **Embedding layer**
- Fixed context window (sequence length = 20)
- LSTM for sequential learning
- Softmax output over vocabulary
- Early stopping to prevent overfitting

---

## 📁 Project Structure
LSTM_next_word_predict/
│
├── app.py                         
├── experiemnts.ipynb              
├── hamlet.txt                     
├── tokenizer.pickle              
├── next_word_lstm.keras           
├── requirements.txt            
├── README.md                      

