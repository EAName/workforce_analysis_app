import pytest
import pandas as pd
import numpy as np
from agents.diversity_agent import monitor_diversity

@pytest.fixture
def sample_data():
    """Create sample HR data for testing diversity analysis"""
    data = {
        'EmployeeNumber': range(1, 6),
        'gender': ['Female', 'Male', 'Female', 'Male', 'Female'],
        'ethnicity': ['Asian', 'White', 'Hispanic', 'Black', 'Asian'],
        'is_leader': [True, False, True, False, True],
        'status': ['Active', 'Left', 'Active', 'Active', 'Left'],
        'salary': [80000, 70000, 75000, 65000, 72000]
    }
    return pd.DataFrame(data)

def test_monitor_diversity(sample_data):
    """Test diversity monitoring function"""
    results = monitor_diversity(sample_data)
    
    # Test gender ratio
    assert 'gender_ratio' in results
    assert isinstance(results['gender_ratio'], float)
    assert 0 <= results['gender_ratio'] <= 1
    
    # Test ethnicity distribution
    assert 'ethnicity_distribution' in results
    assert isinstance(results['ethnicity_distribution'], dict)
    assert sum(results['ethnicity_distribution'].values()) == pytest.approx(1.0)
    
    # Test leadership diversity
    assert 'female_leadership_ratio' in results
    assert isinstance(results['female_leadership_ratio'], float)
    assert 0 <= results['female_leadership_ratio'] <= 1
    
    # Test turnover by gender
    assert 'turnover_by_gender' in results
    assert isinstance(results['turnover_by_gender'], dict)
    assert sum(results['turnover_by_gender'].values()) == pytest.approx(1.0)
    
    # Test pay equity
    assert 'median_salary_by_gender' in results
    assert isinstance(results['median_salary_by_gender'], dict)
    assert 'pay_equity_ratio' in results
    assert isinstance(results['pay_equity_ratio'], float) 