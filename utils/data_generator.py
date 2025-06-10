import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import random
from schemas.data_schema import HR_SCHEMA

class HRDataGenerator:
    """Generate synthetic HR data for testing"""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Define department and role relationships
        self.department_roles = {
            'IT': ['Developer', 'Engineer', 'System Administrator', 'IT Manager', 'Technical Specialist'],
            'HR': ['HR Manager', 'HR Specialist', 'Recruiter', 'HR Director'],
            'Finance': ['Financial Analyst', 'Accountant', 'Finance Manager', 'Controller'],
            'Marketing': ['Marketing Specialist', 'Marketing Manager', 'Brand Manager', 'Marketing Director'],
            'Operations': ['Operations Manager', 'Operations Specialist', 'Supply Chain Manager'],
            'Sales': ['Sales Representative', 'Sales Manager', 'Account Executive', 'Sales Director'],
            'Research': ['Research Scientist', 'Research Analyst', 'Research Director'],
            'Engineering': ['Engineer', 'Senior Engineer', 'Engineering Manager', 'Technical Director']
        }
        
        # Define salary ranges by role
        self.salary_ranges = {
            'Developer': (60000, 120000),
            'Engineer': (65000, 130000),
            'System Administrator': (55000, 110000),
            'IT Manager': (80000, 150000),
            'Technical Specialist': (70000, 130000),
            'HR Manager': (70000, 130000),
            'HR Specialist': (50000, 90000),
            'Recruiter': (45000, 85000),
            'HR Director': (90000, 160000),
            'Financial Analyst': (55000, 100000),
            'Accountant': (50000, 95000),
            'Finance Manager': (75000, 140000),
            'Controller': (80000, 150000),
            'Marketing Specialist': (50000, 95000),
            'Marketing Manager': (70000, 130000),
            'Brand Manager': (65000, 120000),
            'Marketing Director': (85000, 150000),
            'Operations Manager': (65000, 120000),
            'Operations Specialist': (50000, 90000),
            'Supply Chain Manager': (70000, 130000),
            'Sales Representative': (45000, 85000),
            'Sales Manager': (65000, 120000),
            'Account Executive': (55000, 110000),
            'Sales Director': (80000, 150000),
            'Research Scientist': (70000, 130000),
            'Research Analyst': (55000, 100000),
            'Research Director': (85000, 150000),
            'Senior Engineer': (75000, 140000),
            'Engineering Manager': (80000, 150000),
            'Technical Director': (90000, 160000)
        }
        
        # Define education field probabilities by department
        self.education_field_probs = {
            'IT': {'Technical Degree': 0.6, 'Life Sciences': 0.2, 'Other': 0.2},
            'HR': {'Human Resources': 0.5, 'Life Sciences': 0.2, 'Other': 0.3},
            'Finance': {'Life Sciences': 0.3, 'Marketing': 0.2, 'Other': 0.5},
            'Marketing': {'Marketing': 0.6, 'Life Sciences': 0.2, 'Other': 0.2},
            'Operations': {'Technical Degree': 0.4, 'Life Sciences': 0.3, 'Other': 0.3},
            'Sales': {'Marketing': 0.4, 'Life Sciences': 0.2, 'Other': 0.4},
            'Research': {'Life Sciences': 0.7, 'Medical': 0.2, 'Other': 0.1},
            'Engineering': {'Technical Degree': 0.7, 'Life Sciences': 0.2, 'Other': 0.1}
        }
    
    def generate_data(self, n_employees: int = 1000, start_date: str = '2010-01-01') -> pd.DataFrame:
        """Generate synthetic HR data"""
        data = {
            'EmployeeNumber': range(1, n_employees + 1),
            'Age': self._generate_ages(n_employees),
            'Department': self._generate_departments(n_employees),
            'JobRole': [],  # Will be filled based on department
            'Salary': [],  # Will be filled based on role
            'YearsAtCompany': [],  # Will be filled based on hire date
            'JobSatisfaction': np.random.randint(1, 6, n_employees),
            'WorkLifeBalance': np.random.randint(1, 6, n_employees),
            'PerformanceRating': np.random.randint(1, 6, n_employees),
            'Education': np.random.randint(1, 6, n_employees),
            'EducationField': [],  # Will be filled based on department
            'Gender': np.random.choice(['Male', 'Female'], n_employees, p=[0.6, 0.4]),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_employees, p=[0.3, 0.5, 0.2]),
            'NumCompaniesWorked': np.random.randint(0, 11, n_employees),
            'TotalWorkingYears': [],  # Will be calculated
            'TrainingTimesLastYear': np.random.randint(0, 7, n_employees),
            'YearsInCurrentRole': [],  # Will be calculated
            'YearsSinceLastPromotion': [],  # Will be calculated
            'YearsWithCurrManager': [],  # Will be calculated
            'HireDate': [],  # Will be generated
            'TerminationDate': [],  # Will be generated
            'Attrition': []  # Will be generated based on other factors
        }
        
        # Generate hire dates
        start_date = pd.to_datetime(start_date)
        hire_dates = [start_date + timedelta(days=np.random.randint(0, 3650)) for _ in range(n_employees)]
        data['HireDate'] = sorted(hire_dates)
        
        # Generate departments and roles
        for i in range(n_employees):
            dept = data['Department'][i]
            role = np.random.choice(self.department_roles[dept])
            data['JobRole'].append(role)
            data['Salary'].append(np.random.randint(*self.salary_ranges[role]))
            
            # Generate education field based on department
            probs = self.education_field_probs[dept]
            data['EducationField'].append(np.random.choice(list(probs.keys()), p=list(probs.values())))
        
        # Calculate years at company
        current_date = datetime.now()
        data['YearsAtCompany'] = [int((current_date - pd.to_datetime(d)).days / 365.25) for d in data['HireDate']]
        
        # Calculate total working years (years at company + previous experience)
        data['TotalWorkingYears'] = [int(y + np.random.randint(0, 10)) for y in data['YearsAtCompany']]
        
        # Calculate years in current role (less than years at company)
        data['YearsInCurrentRole'] = [int(min(y, np.random.randint(1, int(y * 2) + 1))) for y in data['YearsAtCompany']]
        
        # Calculate years since last promotion (less than years in current role)
        data['YearsSinceLastPromotion'] = [int(min(y, np.random.randint(0, int(y * 1.5) + 1))) for y in data['YearsInCurrentRole']]
        
        # Calculate years with current manager (less than years at company)
        data['YearsWithCurrManager'] = [int(min(y, np.random.randint(1, int(y * 1.5) + 1))) for y in data['YearsAtCompany']]
        
        # Generate termination dates and attrition
        for i in range(n_employees):
            # Calculate attrition probability based on various factors
            attrition_prob = self._calculate_attrition_probability(
                data['YearsAtCompany'][i],
                data['JobSatisfaction'][i],
                data['WorkLifeBalance'][i],
                data['PerformanceRating'][i],
                data['YearsSinceLastPromotion'][i]
            )
            
            if np.random.random() < attrition_prob:
                # Employee left
                termination_date = data['HireDate'][i] + timedelta(days=np.random.randint(365, 3650))
                data['TerminationDate'].append(termination_date)
                data['Attrition'].append('Yes')
            else:
                # Employee still working
                data['TerminationDate'].append(pd.NaT)
                data['Attrition'].append('No')
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Validate against schema
        HR_SCHEMA.validate_dataframe(df)
        
        return df
    
    def _generate_ages(self, n: int) -> np.ndarray:
        """Generate realistic age distribution"""
        # Generate ages with a normal distribution centered around 35
        ages = np.random.normal(35, 8, n)
        # Clip to realistic range (18-65)
        return np.clip(ages, 18, 65).astype(int)
    
    def _generate_departments(self, n: int) -> np.ndarray:
        """Generate department distribution"""
        departments = list(self.department_roles.keys())
        # IT and Engineering are more common
        probs = [0.25, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        return np.random.choice(departments, n, p=probs)
    
    def _calculate_attrition_probability(self, years_at_company: float, 
                                      job_satisfaction: int,
                                      work_life_balance: int,
                                      performance_rating: int,
                                      years_since_promotion: float) -> float:
        """Calculate attrition probability based on various factors"""
        # Base probability
        prob = 0.1
        
        # Adjust based on years at company (U-shaped curve)
        if years_at_company < 1:
            prob += 0.2  # High turnover in first year
        elif years_at_company > 5:
            prob += 0.1  # Higher turnover after 5 years
        
        # Adjust based on job satisfaction
        prob += (5 - job_satisfaction) * 0.05
        
        # Adjust based on work-life balance
        prob += (5 - work_life_balance) * 0.05
        
        # Adjust based on performance rating
        if performance_rating < 3:
            prob += 0.1  # Higher turnover for low performers
        
        # Adjust based on years since promotion
        if years_since_promotion > 3:
            prob += 0.1  # Higher turnover if no promotion in 3+ years
        
        # Ensure probability is between 0 and 1
        return min(max(prob, 0), 1)

def generate_test_data(n_employees: int = 1000, output_file: Optional[str] = None) -> pd.DataFrame:
    """Generate test data and optionally save to file"""
    generator = HRDataGenerator()
    df = generator.generate_data(n_employees)
    
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Generated data saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Generate sample data when run directly
    df = generate_test_data(1000, "data/synthetic_hr_data.csv")
    print(f"Generated {len(df)} employee records")
    print("\nSample data:")
    print(df.head())
    print("\nData summary:")
    print(df.describe()) 