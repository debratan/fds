import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os
from tensorflow.keras.models import load_model as tf_load_model

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("index.html")

def load_model(filename):
    # Get the absolute path of the current file
    base_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "model")  # Construct the absolute path to the 'model' directory
    filepath = os.path.join(model_dir, filename)  # Construct the full path to the model file

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Load TensorFlow model
    model = tf_load_model(filepath)
    return model, None  # Return None for scaler if not applicable

def load_message_model(filename):
    # Get the absolute path of the current file
    base_dir = os.path.abspath(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "model")  # Construct the absolute path to the 'model' directory
    model_path = os.path.join(model_dir, filename)
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    # Load TensorFlow model
    model = tf_load_model(model_path)

    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def contains_suspicious_words(message, keyword_list):
    message = message.lower()
    return int(any(keyword in message for keyword in keyword_list))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Load models
        fraud_model, _ = load_model("anomaly_model.h5")
        message_model, vectorizer = load_message_model("message_model.h5")

        # Load scaler for fraud_model
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        # Validate input
        if 'features' not in data or not isinstance(data['features'], dict):
            return jsonify({"error": "Invalid input format"}), 400

        # Extract features
        features = data['features']
        amount = features.get('amount', 0)
        transaction_type = features.get('transaction_type', 0)
        account_id = features.get('account_id', 0)
        message = features.get('message', "")

        # Step 1: Predict suspicious_flag probability using message_model
        message_vector = vectorizer.transform([message])  # Use the loaded vectorizer
        suspicious_flag_prob = float(message_model.predict(message_vector)[0][0])  # TensorFlow model output

        # Step 2: Predict fraud probability using transaction_model
        transaction_features = pd.DataFrame([[amount, transaction_type, account_id, suspicious_flag_prob]],
                                            columns=['amount', 'transaction_type', 'account_id', 'suspicious_flag'])
        print("Transaction Features for fraud_model:", transaction_features)

        # Scale the features
        transaction_features_scaled = scaler.transform(transaction_features)

        # Predict fraud probability
        fraud_prob = float(fraud_model.predict(transaction_features_scaled)[0][0])  # TensorFlow model output
        

        # Combine probabilities with new weights (0.6 for message_model, 0.4 for fraud_model)
        combined_score = float((0.6 * suspicious_flag_prob) + (0.4 * fraud_prob))
        is_fraud = combined_score >= 0.5  # Threshold for classification

        # Combine results
        result = {
            "is_fraud": bool(is_fraud),
            "suspicious_flag_probability": suspicious_flag_prob,
            "fraud_probability": fraud_prob,
            "combined_score": combined_score
        }

        return jsonify(result)

    except Exception as e:
        print("Error occurred:", str(e))
        return jsonify({"error": "Error occurred while processing the request.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
