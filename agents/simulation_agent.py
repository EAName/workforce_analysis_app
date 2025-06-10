import pandas as pd
import numpy as np

def simulate_attrition_interventions(df: pd.DataFrame,
                                    intervention: dict,
                                    participation_rate: float = 1.0):
    """
    Simulate retention what-if scenarios by reducing attrition risk for participants.

    Parameters:
    - df: DataFrame with columns ['employee_id', 'attrited'] plus risk features.
    - intervention: {
          'type': str,
          'effect_size_pct': float,    # % reduction in individual risk
          'cost_per_employee': float
      }
    - participation_rate: fraction of workforce included in intervention [0.0–1.0].

    Returns:
    - result: dict with projected rates, saved headcount, and intervention cost.
    """
    # Copy and ensure attrited is numeric 0/1
    data = df.copy()
    data['attrited'] = data['attrited'].astype(int)

    # Baseline metrics
    total = len(data)
    baseline_attritions = data['attrited'].sum()
    baseline_attrition_rate = baseline_attritions / total
    baseline_retention_rate = 1 - baseline_attrition_rate

    # Determine participants
    n_participants = int(total * participation_rate)
    # Simple random assignment of participants
    participants = data.sample(n=n_participants, random_state=42).index
    data['is_participant'] = False
    data.loc[participants, 'is_participant'] = True

    # Apply risk reduction: reduce attrition probability for participants
    # Here we assume attrited==1 means they would leave without intervention.
    # We “rescue” a fraction = effect_size_pct of those cases.
    effect = intervention.get('effect_size_pct', 0) / 100.0

    # Count rescued attritions among participants
    part_attrited = data.loc[data['is_participant'] & (data['attrited']==1)]
    rescued = int(len(part_attrited) * effect)

    # Projected attritions = baseline attritions – rescued
    projected_attritions = baseline_attritions - rescued
    projected_attrition_rate = projected_attritions / total
    projected_retention_rate = 1 - projected_attrition_rate

    # Calculate intervention cost
    cost_per_emp = intervention.get('cost_per_employee', 0)
    total_cost = n_participants * cost_per_emp

    result = {
        'baseline_attrition_rate': round(baseline_attrition_rate, 3),
        'projected_attrition_rate': round(projected_attrition_rate, 3),
        'baseline_retention_rate': round(baseline_retention_rate, 3),
        'projected_retention_rate': round(projected_retention_rate, 3),
        'employees_participating': n_participants,
        'attritions_rescued': rescued,
        'intervention_cost': round(total_cost, 2),
        'intervention_type': intervention.get('type')
    }

    return result
