import pytest
from fastapi.testclient import TestClient

def test_simulate_attrition(client: TestClient, sample_employee_data):
    """Test attrition simulation endpoint"""
    response = client.post(
        "/api/simulation/attrition",
        json={
            "data": sample_employee_data,
            "intervention_impact": 0.2,
            "time_horizon": 12
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "baseline_attrition" in data
    assert "projected_attrition" in data
    assert "improvement" in data
    assert "monthly_projections" in data
    assert "cost_savings" in data
    
    # Check data types
    assert isinstance(data["baseline_attrition"], float)
    assert isinstance(data["projected_attrition"], float)
    assert isinstance(data["improvement"], float)
    assert isinstance(data["monthly_projections"], list)
    assert isinstance(data["cost_savings"], float)
    
    # Check values
    assert 0 <= data["baseline_attrition"] <= 1
    assert 0 <= data["projected_attrition"] <= 1
    assert data["improvement"] >= 0
    assert data["cost_savings"] >= 0
    
    # Check monthly projections
    assert len(data["monthly_projections"]) == 12
    for projection in data["monthly_projections"]:
        assert "month" in projection
        assert "attrition_rate" in projection
        assert "employees_at_risk" in projection
        assert isinstance(projection["month"], int)
        assert isinstance(projection["attrition_rate"], float)
        assert isinstance(projection["employees_at_risk"], int)

def test_get_interventions(client: TestClient):
    """Test interventions endpoint"""
    response = client.get("/api/simulation/interventions")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert isinstance(data, dict)
    assert len(data) > 0
    
    # Check values
    for intervention, impact in data.items():
        assert isinstance(intervention, str)
        assert "min_impact" in impact
        assert "max_impact" in impact
        assert "typical_impact" in impact
        assert 0 <= impact["min_impact"] <= impact["typical_impact"] <= impact["max_impact"] <= 1

def test_simulate_attrition_invalid_data(client: TestClient):
    """Test attrition simulation with invalid data"""
    response = client.post(
        "/api/simulation/attrition",
        json={
            "data": {"invalid": "data"},
            "intervention_impact": 2.0,  # Invalid impact > 1
            "time_horizon": -1  # Invalid time horizon
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_simulate_attrition_missing_data(client: TestClient):
    """Test attrition simulation with missing data"""
    response = client.post(
        "/api/simulation/attrition",
        json={}
    )
    
    assert response.status_code == 422  # Validation error 