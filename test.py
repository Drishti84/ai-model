import requests

url = "https://ai-model-gcra.onrender.com/predict"
headers = {"Content-Type": "application/json"}
data = {"symptoms": ["joint pain", "shivering"]}

response = requests.post(url, json=data, headers=headers)
print(response.json())  # Should return the disease prediction
