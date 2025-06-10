import pytest
import pandas as pd
import numpy as np
from agents.attrition_agent import preprocess_data, predict_attrition, AttritionAgent
from agents.base_agent import BaseAgent
from schemas.data_schema import HR_SCHEMA

@pytest.fixture
def sample_data():
    """Create sample HR data for testing"""
    data = {
        'EmployeeNumber': range(1, 6),
        'Attrition': ['Yes', 'No', 'Yes', 'No', 'No'],
        'Age': [30, 35, 40, 45, 50],
        'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
        'JobRole': ['Developer', 'HR Manager', 'Developer', 'Accountant', 'HR Manager'],
        'Salary': [50000, 60000, 55000, 65000, 70000],
        'YearsAtCompany': [2, 5, 3, 7, 4],
        'JobSatisfaction': [4, 3, 2, 5, 4],
        'WorkLifeBalance': [3, 4, 2, 5, 3],
        'PerformanceRating': [4, 3, 2, 5, 4],
        'Education': [3, 4, 2, 5, 3],
        'EducationField': ['Technical Degree', 'Life Sciences', 'Marketing', 'Medical', 'Other'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
        'MaritalStatus': ['Single', 'Married', 'Divorced', 'Married', 'Single'],
        'NumCompaniesWorked': [2, 1, 3, 1, 2],
        'TotalWorkingYears': [5, 8, 12, 15, 10],
        'TrainingTimesLastYear': [2, 3, 1, 4, 2],
        'YearsInCurrentRole': [1, 3, 2, 4, 2],
        'YearsSinceLastPromotion': [1, 2, 3, 1, 2],
        'YearsWithCurrManager': [1, 2, 1, 3, 2],
        'HireDate': pd.date_range(start='2020-01-01', periods=5, freq='M'),
        'TerminationDate': [pd.NaT, pd.NaT, pd.Timestamp('2023-12-31'), pd.NaT, pd.NaT]
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_data):
    """Test data preprocessing function"""
    processed_data = preprocess_data(sample_data)
    
    # Check if categorical variables are converted to numeric
    assert 'Department_IT' in processed_data.columns
    assert 'Department_HR' in processed_data.columns
    assert 'JobRole_Developer' in processed_data.columns
    assert 'EducationField_Technical Degree' in processed_data.columns
    assert 'Gender_Male' in processed_data.columns
    assert 'MaritalStatus_Single' in processed_data.columns
    
    # Check if numeric columns are preserved
    assert 'Age' in processed_data.columns
    assert 'Salary' in processed_data.columns
    assert 'YearsAtCompany' in processed_data.columns
    assert 'JobSatisfaction' in processed_data.columns
    assert 'WorkLifeBalance' in processed_data.columns
    assert 'PerformanceRating' in processed_data.columns
    assert 'Education' in processed_data.columns
    assert 'NumCompaniesWorked' in processed_data.columns
    assert 'TotalWorkingYears' in processed_data.columns
    assert 'TrainingTimesLastYear' in processed_data.columns
    assert 'YearsInCurrentRole' in processed_data.columns
    assert 'YearsSinceLastPromotion' in processed_data.columns
    assert 'YearsWithCurrManager' in processed_data.columns
    
    # Check if there are no missing values
    assert not processed_data.isnull().any().any()

def test_train_attrition_model(sample_data):
    """Test model training function using AttritionAgent"""
    agent = AttritionAgent()
    model, scaler, feature_columns = agent.train_model(sample_data)
    
    # Check if model is trained
    assert hasattr(model, 'predict_proba')
    
    # Check if scaler is fitted
    assert hasattr(scaler, 'transform')
    
    # Check if feature columns are returned
    assert len(feature_columns) > 0
    
    # Check if all new features are included
    expected_features = [
        'Age', 'Salary', 'YearsAtCompany', 'JobSatisfaction',
        'WorkLifeBalance', 'PerformanceRating', 'Education',
        'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    for feature in expected_features:
        assert any(feature in col for col in feature_columns)

def test_predict_attrition(sample_data):
    """Test prediction function"""
    results = predict_attrition(sample_data)
    
    # Check if results contain required columns
    assert 'EmployeeNumber' in results.columns
    assert 'attrition_risk' in results.columns
    
    # Check if predictions are probabilities
    assert all((results['attrition_risk'] >= 0) & (results['attrition_risk'] <= 1))
    
    # Check if results have correct number of rows
    assert len(results) == len(sample_data)

def test_schema_validation(sample_data):
    """Test schema validation with new columns"""
    # Validate against schema
    assert HR_SCHEMA.validate_dataframe(sample_data)
    
    # Test invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'Age'] = 15  # Below minimum age
    with pytest.raises(ValueError):
        HR_SCHEMA.validate_dataframe(invalid_data)
    
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'Department'] = 'Invalid'  # Invalid department
    with pytest.raises(ValueError):
        HR_SCHEMA.validate_dataframe(invalid_data)
    
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'JobSatisfaction'] = 6  # Above maximum rating
    with pytest.raises(ValueError):
        HR_SCHEMA.validate_dataframe(invalid_data) 