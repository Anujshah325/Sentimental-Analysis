import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model & word index
model = tf.keras.models.load_model("sentiment_lstm.h5")
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

VOCAB_SIZE = 10000
MAX_LEN = 200

def preprocess_text(text):
    text = text.lower().split()
    encoded = [word_index.get(w, 2) for w in text]  # 2 = OOV
    padded = pad_sequences([encoded], maxlen=MAX_LEN)
    return padded

def predict_sentiment(text):
    seq = preprocess_text(text)
    prediction = model.predict(seq, verbose=0)[0][0]
    sentiment = "Positive 😀" if prediction > 0.5 else "Negative 😞"
    return sentiment, float(prediction)

if __name__ == "__main__":
    while True:
        review = input("\nEnter a review (or 'quit'): ")
        if review.lower() == "quit":
            break
        sentiment, score = predict_sentiment(review)
        print(f"Prediction: {sentiment} (Confidence: {score:.2f})")
