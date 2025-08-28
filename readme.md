# Sentiment Analysis using LSTM

Detect sentiment (positive/negative) from text reviews using an LSTM-based model. Optionally, a Streamlit web interface is included for interactive predictions.

---

## 📂 Project Structure

sentiment_analysis_lstm/
│
├─ model_training.py # Train and save LSTM model
├─ app.py # Streamlit app for sentiment prediction
├─ sentiment_lstm.h5 # Trained model
├─ tokenizer.pkl # Tokenizer for text preprocessing
└─ README.md

yaml
Copy code

---

## ⚡ Steps to Run

### 1️⃣ Set up environment
```bash
# Create virtual environment (optional but recommended)
python -m venv ace_env
# Activate (Windows)
ace_env\Scripts\activate
# Install dependencies
pip install tensorflow streamlit
2️⃣ Train the model
bash
Copy code
python model_training.py
This generates sentiment_lstm.h5 and tokenizer.pkl.

3️⃣ Run the Streamlit app
bash
Copy code
streamlit run app.py
Open the local URL shown in the terminal (http://localhost:8501).

Enter any review text to see the predicted sentiment.

📝 Sample Outputs
Terminal Output after Training

yaml
Copy code
Epoch 5/5
391/391 [==============================] - accuracy: 0.9154 - val_accuracy: 0.8484
Model saved to sentiment_lstm.h5
Test Accuracy: 0.8484
Streamlit App

vbnet
Copy code
Input: "I loved this movie, it was fantastic!"
Output: Positive
vbnet
Copy code
Input: "The film was boring and too long."
Output: Negative