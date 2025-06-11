import pytest
import pandas as pd
import numpy as np
from agents.planning_agent import forecast_workforce_plan

@pytest.fixture
def sample_headcount_plan():
    """Create sample headcount plan data"""
    return pd.DataFrame({
        'role': ['Developer', 'Data Scientist', 'Product Manager'],
        'planned_hires': [5, 3, 2],
        'avg_salary': [100000, 120000, 110000]
    })

@pytest.fixture
def sample_hiring_pipeline():
    """Create sample hiring pipeline data"""
    return pd.DataFrame({
        'role': ['Developer', 'Data Scientist', 'Product Manager'],
        'conversion_rate': [0.8, 0.7, 0.9]
    })

def test_forecast_workforce_plan(sample_headcount_plan, sample_hiring_pipeline):
    """Test workforce planning forecast function"""
    results = forecast_workforce_plan(sample_headcount_plan, sample_hiring_pipeline)
    
    # Test results structure
    assert isinstance(results, dict)
    assert 'next_quarter_hires' in results
    assert 'budget_impact' in results
    assert 'by_role' in results
    
    # Test data types
    assert isinstance(results['next_quarter_hires'], int)
    assert isinstance(results['budget_impact'], float)
    assert isinstance(results['by_role'], list)
    
    # Test calculations
    # Expected hires should be less than or equal to planned hires due to conversion rates
    total_planned = sample_headcount_plan['planned_hires'].sum()
    assert results['next_quarter_hires'] <= total_planned
    
    # Budget impact should be positive
    assert results['budget_impact'] > 0
    
    # Test by_role structure
    for role_plan in results['by_role']:
        assert 'role' in role_plan
        assert 'expected_hires' in role_plan
        assert 'total_cost' in role_plan
        
        # Test data types
        assert isinstance(role_plan['role'], str)
        assert isinstance(role_plan['expected_hires'], (int, float))
        assert isinstance(role_plan['total_cost'], (int, float))
        
        # Test calculations
        assert role_plan['expected_hires'] >= 0
        assert role_plan['total_cost'] >= 0 