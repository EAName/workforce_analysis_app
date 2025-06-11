import pytest
import pandas as pd
import numpy as np
from agents.simulation_agent import simulate_attrition_interventions

@pytest.fixture
def sample_data():
    """Create sample HR data for testing attrition simulation"""
    data = {
        'employee_id': range(1, 6),
        'attrited': [1, 0, 1, 0, 1],
        'salary': [80000, 70000, 75000, 65000, 72000],
        'years_at_company': [2, 5, 3, 7, 4],
        'job_satisfaction': [3, 4, 2, 5, 3]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_intervention():
    """Create sample intervention data"""
    return {
        'type': 'Career Development Program',
        'effect_size_pct': 30.0,
        'cost_per_employee': 5000.0
    }

def test_simulate_attrition_interventions(sample_data, sample_intervention):
    """Test attrition intervention simulation function"""
    # Test with full participation
    results_full = simulate_attrition_interventions(
        sample_data,
        sample_intervention,
        participation_rate=1.0
    )
    
    # Test results structure
    assert isinstance(results_full, dict)
    assert 'baseline_attrition_rate' in results_full
    assert 'projected_attrition_rate' in results_full
    assert 'baseline_retention_rate' in results_full
    assert 'projected_retention_rate' in results_full
    assert 'employees_participating' in results_full
    assert 'attritions_rescued' in results_full
    assert 'intervention_cost' in results_full
    assert 'intervention_type' in results_full
    
    # Test data types
    assert isinstance(results_full['baseline_attrition_rate'], float)
    assert isinstance(results_full['projected_attrition_rate'], float)
    assert isinstance(results_full['baseline_retention_rate'], float)
    assert isinstance(results_full['projected_retention_rate'], float)
    assert isinstance(results_full['employees_participating'], int)
    assert isinstance(results_full['attritions_rescued'], int)
    assert isinstance(results_full['intervention_cost'], float)
    assert isinstance(results_full['intervention_type'], str)
    
    # Test calculations
    assert 0 <= results_full['baseline_attrition_rate'] <= 1
    assert 0 <= results_full['projected_attrition_rate'] <= 1
    assert 0 <= results_full['baseline_retention_rate'] <= 1
    assert 0 <= results_full['projected_retention_rate'] <= 1
    assert results_full['employees_participating'] == len(sample_data)
    assert results_full['attritions_rescued'] >= 0
    assert results_full['intervention_cost'] > 0
    
    # Test with partial participation
    results_partial = simulate_attrition_interventions(
        sample_data,
        sample_intervention,
        participation_rate=0.5
    )
    
    # Test that partial participation results in lower costs
    assert results_partial['intervention_cost'] < results_full['intervention_cost']
    assert results_partial['employees_participating'] < results_full['employees_participating']
    
    # Test that intervention type is preserved
    assert results_partial['intervention_type'] == sample_intervention['type'] 