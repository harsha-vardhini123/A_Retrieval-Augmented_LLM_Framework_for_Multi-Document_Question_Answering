import requests
from config import config

API_KEY = config.GEMINI_API_KEY
endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

headers = {
    "Authorization": f"Bearer {API_KEY}",
}

response = requests.get(endpoint, headers=headers)

if response.status_code == 200:
    models = response.json()
    for model in models.get("models", []):
        print(f"Model name: {model['name']}, Supported Methods: {model.get('supportedMethods', [])}")
else:
    print(f"Error: {response.status_code} {response.text}")
