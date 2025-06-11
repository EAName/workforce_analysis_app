import pytest
from fastapi.testclient import TestClient

def test_analyze_skill_gaps(client: TestClient, sample_employee_data, sample_resume_texts, sample_transcripts):
    """Test skill gap analysis endpoint"""
    response = client.post(
        "/api/skill-gap/analyze",
        json={
            "data": sample_employee_data,
            "resume_texts": sample_resume_texts,
            "transcripts": sample_transcripts
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "recommendations" in data
    assert "missing_skills" in data
    assert "training_recommendations" in data
    
    # Check data types
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["missing_skills"], dict)
    assert isinstance(data["training_recommendations"], dict)
    
    # Check values
    assert len(data["recommendations"]) > 0
    for rec in data["recommendations"]:
        assert "employee_id" in rec
        assert "missing_skills" in rec
        assert "recommendations" in rec
        assert isinstance(rec["missing_skills"], list)
        assert isinstance(rec["recommendations"], list)

def test_get_required_skills(client: TestClient):
    """Test required skills endpoint"""
    response = client.get("/api/skill-gap/required-skills")
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert isinstance(data, dict)
    assert len(data) > 0
    
    # Check values
    for role, skills in data.items():
        assert isinstance(role, str)
        assert isinstance(skills, list)
        assert all(isinstance(skill, str) for skill in skills)

def test_analyze_skill_gaps_invalid_data(client: TestClient):
    """Test skill gap analysis with invalid data"""
    response = client.post(
        "/api/skill-gap/analyze",
        json={
            "data": {"invalid": "data"},
            "resume_texts": {},
            "transcripts": {}
        }
    )
    
    assert response.status_code == 422  # Validation error

def test_analyze_skill_gaps_missing_data(client: TestClient):
    """Test skill gap analysis with missing data"""
    response = client.post(
        "/api/skill-gap/analyze",
        json={}
    )
    
    assert response.status_code == 422  # Validation error 