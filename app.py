import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# ---------------- CONFIG ----------------
MODEL_PATH = "sentiment_lstm.h5"
VOCAB_SIZE = 10000
MAX_LEN = 200

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------- LOAD WORD INDEX ----------------
word_index = imdb.get_word_index()

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_text(text):
    # Convert text to integer sequence using IMDb word_index
    seq = [word_index.get(w, 2) + 3 for w in text.lower().split()]  # 2 = OOV token
    seq = sequence.pad_sequences([seq], maxlen=MAX_LEN)
    return seq

# ---------------- STREAMLIT UI ----------------
st.title("🎬 IMDb Sentiment Analysis")
st.write("Enter a movie review and see if it is Positive or Negative!")

user_input = st.text_area("Your review here:")

if st.button("Predict Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text for prediction.")
    else:
        processed_input = preprocess_text(user_input)
        pred = model.predict(processed_input)[0][0]
        sentiment = "Positive ✅" if pred > 0.5 else "Negative ❌"
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {pred:.2f}")
