from flask import Flask, request, jsonify
import torch
import pickle
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from model import DiseasePredictor  # Ensure this matches your model definition

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and encoders
input_size = 131  # Adjust based on dataset
num_classes = 41  # Adjust based on dataset
model = DiseasePredictor(input_size, num_classes).to(device)
model.load_state_dict(torch.load("disease_model.pth", map_location=device))
model.eval()

with open("mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load precaution data
import pandas as pd
precaution_df = pd.read_csv("data/symptom_precaution.csv")
precaution_dict = {row["Disease"]: [row[f"Precaution_{i}"] for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])] for _, row in precaution_df.iterrows()}

@app.route("/predict", methods=["POST"])
def predict():
    req = request.get_json()
    user_text = req.get("queryResult", {}).get("queryText", "").lower()
    
    # Match symptoms from user_text
    symptom_list = [s for s in mlb.classes_ if s in user_text]

    if not symptom_list:
        return jsonify({"fulfillmentText": "I couldn't find any known symptoms in your message."})

    input_vector = mlb.transform([symptom_list])
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    predicted_disease = label_encoder.inverse_transform([predicted_class.cpu().numpy()[0]])[0]
    precautions = precaution_dict.get(predicted_disease, ["No precautions available."])

    response_text = f"You may have *{predicted_disease}*. Here are some precautions: " + ", ".join(precautions)

    return jsonify({"fulfillmentText": response_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
