import pytest
from fastapi.testclient import TestClient

def test_analyze_diversity(client: TestClient, sample_employee_data):
    """Test diversity analysis endpoint"""
    response = client.post(
        "/api/diversity/analyze",
        json={"data": sample_employee_data}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "gender_ratio" in data
    assert "ethnicity_distribution" in data
    assert "female_leadership_ratio" in data
    assert "turnover_by_gender" in data
    assert "median_salary_by_gender" in data
    assert "pay_equity_ratio" in data
    
    # Check data types
    assert isinstance(data["gender_ratio"], float)
    assert isinstance(data["ethnicity_distribution"], dict)
    assert isinstance(data["female_leadership_ratio"], float)
    assert isinstance(data["turnover_by_gender"], dict)
    assert isinstance(data["median_salary_by_gender"], dict)
    assert isinstance(data["pay_equity_ratio"], float)
    
    # Check values
    assert 0 <= data["gender_ratio"] <= 1
    assert 0 <= data["female_leadership_ratio"] <= 1
    assert 0 <= data["pay_equity_ratio"] <= 2  # Pay equity ratio can be up to 2
    assert abs(sum(data["ethnicity_distribution"].values()) - 1.0) < 0.0001
    assert abs(sum(data["turnover_by_gender"].values()) - 1.0) < 0.0001

def test_get_diversity_metrics(client: TestClient):
    """Test diversity metrics endpoint"""
    response = client.get("/api/diversity/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert isinstance(data, dict)
    assert "message" in data

def test_analyze_diversity_invalid_data(client: TestClient):
    """Test diversity analysis with invalid data"""
    response = client.post(
        "/api/diversity/analyze",
        json={"data": {"invalid": "data"}}
    )
    
    assert response.status_code == 422  # Validation error

def test_analyze_diversity_missing_data(client: TestClient):
    """Test diversity analysis with missing data"""
    response = client.post(
        "/api/diversity/analyze",
        json={}
    )
    
    assert response.status_code == 422  # Validation error 