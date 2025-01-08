from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
from preprocessing import preprocess_text

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('model/sentiment_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        if 'text' not in input_data:
            return jsonify({"error": "Missing 'text' field in the input."}), 400

        # Preprocess the input text
        raw_text = input_data['text']
        cleaned_text = preprocess_text(raw_text)

        # Convert text to feature vector
        text_vector = vectorizer.transform([cleaned_text])

        # Make prediction
        prediction = model.predict(text_vector)

        # Send response
        return jsonify({
            "text": raw_text,
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route('/', methods=['GET'])
def home():
    return "Sentiment Analysis API is running. Use the /predict endpoint to send POST requests with text."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
