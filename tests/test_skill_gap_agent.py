import pytest
import pandas as pd
import numpy as np
from agents.skill_gap_agent import analyze_skill_gap

@pytest.fixture
def sample_data():
    """Create sample HR data for testing skill gap analysis"""
    data = {
        'employee_id': range(1, 6),
        'role': ['Developer', 'Data Scientist', 'Developer', 'Data Scientist', 'Developer'],
        'required_skills': [
            ['Python', 'SQL', 'Machine Learning'],
            ['Python', 'R', 'Statistics'],
            ['Python', 'SQL', 'Machine Learning'],
            ['Python', 'R', 'Statistics'],
            ['Python', 'SQL', 'Machine Learning']
        ]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_resume_texts():
    """Create sample resume texts"""
    return {
        1: "Experienced in Python and SQL",
        2: "Proficient in Python and R",
        3: "Python developer with SQL experience",
        4: "Data scientist with Python and R skills",
        5: "Python and SQL developer"
    }

@pytest.fixture
def sample_transcripts():
    """Create sample training transcripts"""
    return {
        1: ["Python Basics", "SQL Fundamentals"],
        2: ["Python for Data Science", "R Programming"],
        3: ["Python Development", "SQL Advanced"],
        4: ["Python Data Analysis", "R Statistics"],
        5: ["Python Programming", "SQL Database"]
    }

@pytest.fixture
def sample_skill_course_map():
    """Create sample skill to course mapping"""
    return {
        "Python": "Python Programming Course",
        "SQL": "SQL Database Management",
        "Machine Learning": "ML Fundamentals",
        "R": "R Programming",
        "Statistics": "Statistical Analysis"
    }

def test_analyze_skill_gap(sample_data, sample_resume_texts, sample_transcripts, sample_skill_course_map):
    """Test skill gap analysis function"""
    results = analyze_skill_gap(
        sample_data,
        sample_resume_texts,
        sample_transcripts,
        sample_skill_course_map
    )
    
    # Test results structure
    assert isinstance(results, list)
    assert len(results) == len(sample_data)
    
    # Test each recommendation
    for rec in results:
        assert 'employee_id' in rec
        assert 'missing_skills' in rec
        assert 'recommendations' in rec
        
        # Test data types
        assert isinstance(rec['employee_id'], int)
        assert isinstance(rec['missing_skills'], list)
        assert isinstance(rec['recommendations'], list)
        
        # Test that recommendations are not empty
        assert len(rec['recommendations']) > 0
        
        # Test that recommendations match missing skills
        assert len(rec['recommendations']) >= len(rec['missing_skills']) 