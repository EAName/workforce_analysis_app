import pytest
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

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
        'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Operations'], n_employees),
        'JobRole': np.random.choice(['Developer', 'Manager', 'Analyst', 'Designer', 'Consultant'], n_employees),
        'Salary': np.random.randint(40000, 120000, n_employees),
        'YearsAtCompany': np.random.randint(0, 20, n_employees),
        'JobSatisfaction': np.random.randint(1, 5, n_employees),
        'WorkLifeBalance': np.random.randint(1, 5, n_employees),
        'PerformanceRating': np.random.randint(1, 5, n_employees)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_model_dir(tmp_path):
    """Create a temporary directory for model files"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir 