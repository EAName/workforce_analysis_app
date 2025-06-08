import pytest
import pandas as pd
import numpy as np
from agents.attrition_agent import preprocess_data, train_attrition_model, predict_attrition

@pytest.fixture
def sample_data():
    """Create sample HR data for testing"""
    data = {
        'EmployeeNumber': range(1, 6),
        'Attrition': ['Yes', 'No', 'Yes', 'No', 'No'],
        'Age': [30, 35, 40, 45, 50],
        'Department': ['IT', 'HR', 'IT', 'Finance', 'HR'],
        'JobRole': ['Developer', 'Manager', 'Developer', 'Analyst', 'Manager'],
        'Salary': [50000, 60000, 55000, 65000, 70000]
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_data):
    """Test data preprocessing function"""
    processed_data = preprocess_data(sample_data)
    
    # Check if categorical variables are converted to numeric
    assert 'Department_IT' in processed_data.columns
    assert 'Department_HR' in processed_data.columns
    assert 'JobRole_Developer' in processed_data.columns
    
    # Check if numeric columns are preserved
    assert 'Age' in processed_data.columns
    assert 'Salary' in processed_data.columns
    
    # Check if there are no missing values
    assert not processed_data.isnull().any().any()

def test_train_attrition_model(sample_data):
    """Test model training function"""
    model, scaler, feature_columns = train_attrition_model(sample_data)
    
    # Check if model is trained
    assert hasattr(model, 'predict_proba')
    
    # Check if scaler is fitted
    assert hasattr(scaler, 'transform')
    
    # Check if feature columns are returned
    assert len(feature_columns) > 0

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