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