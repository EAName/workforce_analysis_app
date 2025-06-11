import pandas as pd

def monitor_diversity(df):
    """Monitor diversity metrics from employee data"""
    kpis = {}

    # Total counts
    total = len(df)
    total_female = len(df[df['Gender'].str.lower() == 'female'])
    total_leaders = len(df[df['JobRole'].str.contains('Manager|Director|VP|C-Level', case=False, na=False)])

    # Gender diversity
    kpis['gender_ratio'] = total_female / total if total else None

    # Education field distribution (as a proxy for diversity)
    kpis['education_field_distribution'] = df['EducationField'].value_counts(normalize=True).to_dict()

    # Leadership diversity
    female_leaders = len(df[(df['Gender'].str.lower() == 'female') & 
                           (df['JobRole'].str.contains('Manager|Director|VP|C-Level', case=False, na=False))])
    kpis['female_leadership_ratio'] = female_leaders / total_leaders if total_leaders else None

    # Turnover by gender
    left = df[df['Attrition'] == 'Yes']
    kpis['turnover_by_gender'] = left['Gender'].str.lower().value_counts(normalize=True).to_dict()

    # Median salary by gender
    salary_by_gender = df.groupby(df['Gender'].str.lower())['Salary'].median().to_dict()
    kpis['median_salary_by_gender'] = salary_by_gender
    if 'male' in salary_by_gender and 'female' in salary_by_gender:
        kpis['pay_equity_ratio'] = salary_by_gender['female'] / salary_by_gender['male']
    else:
        kpis['pay_equity_ratio'] = None

    # Additional metrics
    kpis['education_level_distribution'] = df['Education'].value_counts(normalize=True).to_dict()
    kpis['marital_status_distribution'] = df['MaritalStatus'].value_counts(normalize=True).to_dict()
    kpis['department_distribution'] = df['Department'].value_counts(normalize=True).to_dict()

    return kpis
