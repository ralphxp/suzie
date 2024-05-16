# app.py
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
model = joblib.load('model.joblib')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    tfidf_text = tfidf_vectorizer.transform([text])
    prediction = model.predict(tfidf_text)[0]
    return render_template('result.html', prediction=prediction, text=text)


if __name__ == '__main__':
    app.run(debug=True)