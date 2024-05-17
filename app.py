
from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

app = Flask(__name__)

# Load pre-trained BERT model and tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# model.eval()

# Function to process text and make predictions
# def predict(text):
#     inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
#     outputs = model(**inputs)
#     logits = outputs.logits
#     probabilities = torch.softmax(logits, dim=1).detach().numpy()
#     return np.argmax(probabilities)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_text():
    text = request.form['text']
    # prediction = predict(text)
    # result = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    return render_template('result.html', text=text, prediction="result")

if __name__ == '__main__':
    app.run(debug=True)
