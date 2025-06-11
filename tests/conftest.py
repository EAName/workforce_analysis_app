import pytest
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(autouse=True)
def load_env():
    """Load environment variables for testing"""
    load_dotenv()

@pytest.fixture
def mock_openai_client(mocker):
    """Mock OpenAI client for testing"""
    return mocker.patch('openai.OpenAI')

@pytest.fixture
def sample_hr_data():
    """Create a larger sample HR dataset for testing"""
    np.random.seed(42)
    n_employees = 100
    
    data = {
        'EmployeeNumber': range(1, n_employees + 1),
        'Attrition': np.random.choice(['Yes', 'No'], n_employees),
        'Age': np.random.randint(25, 65, n_employees),
        'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Operations', 'Sales', 'Research', 'Engineering'], n_employees),
        'JobRole': np.random.choice(['Developer', 'Manager', 'Analyst', 'Designer', 'Consultant', 'Engineer', 'Scientist', 'Specialist', 'Director', 'Executive'], n_employees),
        'Salary': np.random.randint(40000, 120000, n_employees),
        'YearsAtCompany': np.random.randint(0, 20, n_employees),
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'WorkLifeBalance': np.random.randint(1, 5, n_employees),
        'PerformanceRating': np.random.randint(1, 5, n_employees),
        'Education': np.random.randint(1, 5, n_employees),
        'EducationField': np.random.choice(['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'], n_employees),
        'Gender': np.random.choice(['Male', 'Female'], n_employees),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_employees),
        'NumCompaniesWorked': np.random.randint(0, 10, n_employees),
        'TotalWorkingYears': np.random.randint(0, 40, n_employees),
        'TrainingTimesLastYear': np.random.randint(0, 6, n_employees),
        'YearsInCurrentRole': np.random.randint(0, 15, n_employees),
        'YearsSinceLastPromotion': np.random.randint(0, 10, n_employees),
        'YearsWithCurrManager': np.random.randint(0, 15, n_employees),
        'HireDate': pd.date_range(start='2010-01-01', periods=n_employees, freq='M'),
        'TerminationDate': [pd.NaT if x == 'No' else pd.Timestamp('2023-12-31') for x in np.random.choice(['Yes', 'No'], n_employees)]
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
def sample_employee_data():
    """Sample employee data for testing"""
    return {
        "employee_id": ["E001", "E002", "E003", "E004", "E005"],
        "age": [30, 35, 28, 42, 31],
        "gender": ["M", "F", "M", "F", "M"],
        "ethnicity": ["Asian", "White", "Hispanic", "Black", "Asian"],
        "department": ["Engineering", "Marketing", "Sales", "Engineering", "HR"],
        "role": ["Software Engineer", "Marketing Manager", "Sales Rep", "Data Scientist", "HR Manager"],
        "salary": [120000, 95000, 85000, 130000, 90000],
        "tenure": [3, 5, 2, 7, 4],
        "performance_rating": [4.5, 4.0, 3.5, 4.8, 4.2],
        "attrition": [0, 1, 0, 0, 1]
    }

@pytest.fixture
def sample_headcount_plan():
    """Sample headcount plan data for testing"""
    return {
        "role": ["Software Engineer", "Data Scientist", "Product Manager"],
        "planned_hires": [5, 3, 2],
        "average_salary": [120000, 130000, 110000]
    }

@pytest.fixture
def sample_hiring_pipeline():
    """Sample hiring pipeline data for testing"""
    return {
        "role": ["Software Engineer", "Data Scientist", "Product Manager"],
        "candidates": [15, 8, 6],
        "conversion_rate": [0.33, 0.375, 0.33]
    }

@pytest.fixture
def sample_resume_texts():
    """Sample resume texts for testing"""
    return {
        "E001": "Experienced software engineer with Python and Java skills",
        "E002": "Marketing professional with 5 years of experience",
        "E003": "Sales representative with strong communication skills"
    }

@pytest.fixture
def sample_transcripts():
    """Sample training transcripts for testing"""
    return {
        "E001": "Completed Python Advanced Course, Machine Learning Basics",
        "E002": "Completed Digital Marketing Certification",
        "E003": "Completed Sales Techniques Workshop"
    } 