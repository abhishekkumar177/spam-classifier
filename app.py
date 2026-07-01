from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from nltk_setup import download_nltk_resources
from preprocessing import TextPreprocessor, preprocess_text  # needed for unpickling

# Download NLTK data on startup
download_nltk_resources()

app = FastAPI(title="Spam Classifier API")

# Load the trained pipeline once, at startup — not per request!
model = joblib.load("model/spam_classifier_pipeline.pkl")

class Message(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Spam Classifier API is running"}

@app.post("/predict")
def predict(message: Message):
    prediction = model.predict([message.text])[0]
    probability = model.predict_proba([message.text])[0]
    label = "spam" if prediction == 1 else "ham"
    confidence = float(probability[prediction])
    return {
        "message": message.text,
        "prediction": label,
        "confidence": round(confidence, 4)
    }
