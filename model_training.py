import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Config
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5
MODEL_PATH = "sentiment_lstm.h5"

def load_and_preprocess_data():
    """Load IMDb dataset and preprocess sequences."""
    print("📥 Loading IMDb dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")

    # Pad sequences to same length
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    return (x_train, y_train), (x_test, y_test)

def build_model():
    """Builds the LSTM sentiment analysis model."""
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.summary()
    return model

def main():
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_model()

    print("🟢 Training model...")
    model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    print("💾 Saving model...")
    model.save(MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"🎯 Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
