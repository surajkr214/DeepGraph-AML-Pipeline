import requests

# REPLACE with your actual URL
url = "https://deepgraph-aml-pipeline.onrender.com/predict_aml_risk"

payload = {
    "source_account": "Account_A",
    "target_account": "Account_B",
    "transaction_amount": 900
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)