import pytest
from fastapi.testclient import TestClient

def test_forecast_workforce(client: TestClient, sample_headcount_plan, sample_hiring_pipeline):
    """Test workforce forecasting endpoint"""
    response = client.post(
        "/api/planning/forecast",
        json={
            "headcount_plan": sample_headcount_plan,
            "hiring_pipeline": sample_hiring_pipeline
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "next_quarter_hires" in data
    assert "budget_impact" in data
    assert "by_role" in data
    
    # Check data types
    assert isinstance(data["next_quarter_hires"], int)
    assert isinstance(data["budget_impact"], float)
    assert isinstance(data["by_role"], list)
    
    # Check values
    assert data["next_quarter_hires"] > 0
    assert data["budget_impact"] > 0
    assert len(data["by_role"]) > 0
    
    for role_data in data["by_role"]:
        assert "role" in role_data
        assert "expected_hires" in role_data
        assert "total_cost" in role_data
        assert isinstance(role_data["expected_hires"], int)
        assert isinstance(role_data["total_cost"], float)

def test_get_roles(client: TestClient):
    """Test roles endpoint"""
    response = client.get("/api/planning/roles")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert isinstance(data, list)
    assert len(data) > 0
    
    # Check values
    assert all(isinstance(role, str) for role in data)

def test_get_salary_ranges(client: TestClient):
    """Test salary ranges endpoint"""
    response = client.get("/api/planning/salary-ranges")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert isinstance(data, dict)
    assert len(data) > 0
    
    # Check values
    for role, ranges in data.items():
        assert isinstance(role, str)
        assert "min" in ranges
        assert "max" in ranges
        assert "average" in ranges
        assert ranges["min"] <= ranges["average"] <= ranges["max"]

def test_forecast_workforce_invalid_data(client: TestClient):
    """Test workforce forecasting with invalid data"""
    response = client.post(
        "/api/planning/forecast",
        json={
            "headcount_plan": {"invalid": "data"},
            "hiring_pipeline": {"invalid": "data"}
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_forecast_workforce_missing_data(client: TestClient):
    """Test workforce forecasting with missing data"""
    response = client.post(
        "/api/planning/forecast",
        json={}
    )
    
    assert response.status_code == 422  # Validation error 