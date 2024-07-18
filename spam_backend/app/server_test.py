import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_read_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {'message': 'Spam Model Classifier API'}

def test_predict():
    email = {"email": "Congratulations! You've won a free ticket."}
    response = client.post('/predict', json=email)
    assert response.status_code == 200
    assert response.json() in ["Ham", "Spam"]

    email = {"email": "Hello, how are you doing today?"}
    response = client.post('/predict', json=email)
    assert response.status_code == 200
    assert response.json() in ["Ham", "Spam"]
