import pandas as pd

def forecast_workforce_plan(headcount_plan: pd.DataFrame, hiring_pipeline: pd.DataFrame):

    forecast = {}

    # Average conversion rate per role
    avg_conversion = hiring_pipeline.groupby('role')['conversion_rate'].mean()

    # Merge with headcount plan
    merged = headcount_plan.merge(avg_conversion, on='role', how='left')
    merged['conversion_rate'] = merged['conversion_rate'].fillna(1.0)
    merged['expected_hires'] = merged['planned_hires'] * merged['conversion_rate']

    # Calculate cost per hire with 30% overhead
    merged['cost_per_hire'] = merged['avg_salary'] * 1.3
    merged['total_cost'] = merged['expected_hires'] * merged['cost_per_hire']

    # Summarize outputs
    forecast['next_quarter_hires'] = int(merged['expected_hires'].sum())
    forecast['budget_impact'] = round(merged['total_cost'].sum(), 2)
    forecast['by_role'] = merged[['role', 'expected_hires', 'total_cost']].to_dict(orient='records')

    return forecast
