import pandas as pd

def monitor_diversity(df):
    
    kpis = {}

    # Total counts
    total = len(df)
    total_female = len(df[df['gender'] == 'Female'])
    total_leaders = len(df[df['is_leader'] == True])

    # Gender diversity
    kpis['gender_ratio'] = total_female / total if total else None

    # Ethnicity distribution
    kpis['ethnicity_distribution'] = df['ethnicity'].value_counts(normalize=True).to_dict()

    # Leadership diversity
    female_leaders = len(df[(df['gender'] == 'Female') & (df['is_leader'] == True)])
    kpis['female_leadership_ratio'] = female_leaders / total_leaders if total_leaders else None

    # Turnover by gender
    left = df[df['status'] == 'Left']
    kpis['turnover_by_gender'] = left['gender'].value_counts(normalize=True).to_dict()

    # Median salary by gender
    salary_by_gender = df.groupby('gender')['salary'].median().to_dict()
    kpis['median_salary_by_gender'] = salary_by_gender
    if 'Male' in salary_by_gender and 'Female' in salary_by_gender:
        kpis['pay_equity_ratio'] = salary_by_gender['Female'] / salary_by_gender['Male']
    else:
        kpis['pay_equity_ratio'] = None

    return kpis
