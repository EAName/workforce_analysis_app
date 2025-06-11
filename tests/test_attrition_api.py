import pytest
from fastapi.testclient import TestClient

def test_analyze_attrition(client: TestClient, sample_employee_data):
    """Test attrition analysis endpoint"""
    response = client.post(
        "/api/attrition/analyze",
        json={"data": sample_employee_data}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "risk_scores" in data
    assert "high_risk_employees" in data
    assert "metrics" in data
    assert "feature_importance" in data
    
    # Check data types
    assert isinstance(data["risk_scores"], list)
    assert isinstance(data["high_risk_employees"], dict)
    assert isinstance(data["metrics"], dict)
    assert isinstance(data["feature_importance"], dict)
    
    # Check values
    assert len(data["risk_scores"]) == len(sample_employee_data["employee_id"])
    assert all(0 <= score <= 1 for score in data["risk_scores"])

def test_get_feature_importance(client: TestClient):
    """Test feature importance endpoint"""
    response = client.get("/api/attrition/feature-importance")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert isinstance(data, dict)
    assert len(data) > 0
    
    # Check values
    assert all(isinstance(value, float) for value in data.values())
    assert all(0 <= value <= 1 for value in data.values())

def test_analyze_attrition_invalid_data(client: TestClient):
    """Test attrition analysis with invalid data"""
    response = client.post(
        "/api/attrition/analyze",
        json={"data": {"invalid": "data"}}
    )
    
    assert response.status_code == 422  # Validation error

def test_analyze_attrition_missing_data(client: TestClient):
    """Test attrition analysis with missing data"""
    response = client.post(
        "/api/attrition/analyze",
        json={}
    )
    
    assert response.status_code == 422  # Validation error 