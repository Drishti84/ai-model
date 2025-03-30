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
    data = request.json
    symptom_list = data.get("symptoms", [])
    
    valid_symptoms = [s for s in symptom_list if s in mlb.classes_]
    if not valid_symptoms:
        return jsonify({"error": "No valid symptoms found. Please enter correct symptom names."}), 400
    
    input_vector = mlb.transform([valid_symptoms])
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)
    
    predicted_disease = label_encoder.inverse_transform([predicted_class.cpu().numpy()[0]])[0]
    precautions = precaution_dict.get(predicted_disease, ["No precautions available."])
    
    return jsonify({
        "predicted_disease": predicted_disease,
        "precautions": precautions
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
