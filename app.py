from flask import Flask, request, jsonify
import torch
import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from model import DiseasePredictor
import pandas as pd

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and encoders
input_size = 131
num_classes = 41
model = DiseasePredictor(input_size, num_classes).to(device)
model.load_state_dict(torch.load("disease_model.pth", map_location=device))
model.eval()

with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load precaution data
precaution_df = pd.read_csv("data/symptom_precaution.csv")
precaution_dict = {
    row["Disease"]: [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
    for _, row in precaution_df.iterrows()
}

def extract_symptoms_from_text(text):
    # Simple symptom extractor
    possible_symptoms = set([s.lower() for s in mlb.classes_])
    words = text.replace(".", "").replace(",", "").lower().split()
    return [word for word in words if word in possible_symptoms]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    symptom_list = []

    # 1. Structured 'symptoms' (manual API call)
    if "symptoms" in data:
        symptom_list = data["symptoms"]

    # 2. Dialogflow format
    elif "queryResult" in data:
        symptom_list = data["queryResult"].get("parameters", {}).get("symptom", [])

        # 3. Custom payloads in fulfillment
        if not symptom_list:
            for msg in data["queryResult"].get("fulfillmentMessages", []):
                if "payload" in msg and "symptoms" in msg["payload"]:
                    symptom_list = msg["payload"]["symptoms"]
                    break

        # 4. Raw query fallback
        if not symptom_list:
            query_text = data["queryResult"].get("queryText", "").lower()
            symptom_list = extract_symptoms_from_text(query_text)

    print("Incoming raw symptom list:", symptom_list)

    # Normalize and validate
    symptom_list = [s.lower() for s in symptom_list]
    valid_symptoms = [s for s in symptom_list if s in mlb.classes_]

    print("Valid symptoms used for prediction:", valid_symptoms)

    if not valid_symptoms:
        return jsonify({"error": "No valid symptoms found. Please enter correct symptom names."}), 400

    # Prepare model input
    input_vector = mlb.transform([valid_symptoms])
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    predicted_disease = label_encoder.inverse_transform([predicted_class.cpu().numpy()[0]])[0]
    precautions = precaution_dict.get(predicted_disease, ["No precautions available."])

    response_text = f"You may have *{predicted_disease}*. Here are some precautions: " + ", ".join(precautions)

    return jsonify({
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [response_text]
                }
            }
        ]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
