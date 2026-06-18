from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import app

from src.core.config import config

client = TestClient(app)

def test_list_models():
    """Test the /v1/models endpoint."""
    headers = {}
    if config.anthropic_api_key:
        headers = {"x-api-key": config.anthropic_api_key}
    
    response = client.get("/v1/models", headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
    
    # Check if some expected models are present
    model_ids = [model["id"] for model in data["data"]]
    assert "claude-3-5-sonnet-20241022" in model_ids
    assert "claude-3-5-haiku-20241022" in model_ids
    
    # Check model structure
    for model in data["data"]:
        assert "id" in model
        assert "display_name" in model
        assert "created" in model
        assert model["id"].startswith("claude")

def test_root_endpoints():
    """Test the root endpoint to ensure /v1/models is listed."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "models" in data["endpoints"]
    assert data["endpoints"]["models"] == "/v1/models"
