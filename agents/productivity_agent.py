import pandas as pd

def analyze_productivity(time_logs: pd.DataFrame, task_logs: pd.DataFrame, working_hours_per_week: float = 40):
    """
    Surface productivity bottlenecks via analytics.
    
    Parameters:
    - time_logs: DataFrame with ['task_id', 'user_id', 'start_time', 'end_time']
    - task_logs: DataFrame with ['task_id', 'task_type', 'priority', 'created_at', 'completed_at']
    - working_hours_per_week: float, default 40
    
    Returns:
    - insights: dict of KPI metrics and bottleneck signals
    """
    insights = {}

    # --- 1. Cycle times & average completion time ---
    tl = time_logs.copy()
    tl['start_time'] = pd.to_datetime(tl['start_time'])
    tl['end_time']   = pd.to_datetime(tl['end_time'])
    tl['cycle_time_h'] = (tl['end_time'] - tl['start_time']).dt.total_seconds() / 3600.0

    insights['average_completion_time_h'] = tl['cycle_time_h'].mean()

    # Avg cycle time by task type
    merged = tl.merge(task_logs[['task_id', 'task_type']], on='task_id', how='left')
    insights['avg_cycle_time_by_type_h'] = (
        merged.groupby('task_type')['cycle_time_h']
              .mean()
              .to_dict()
    )

    # --- 2. Throughput (tasks completed per week) ---
    tl_completed = task_logs.dropna(subset=['completed_at']).copy()
    tl_completed['completed_at'] = pd.to_datetime(tl_completed['completed_at'])
    tl_completed.set_index('completed_at', inplace=True)
    throughput = (
        tl_completed['task_id']
        .resample('W')
        .nunique()
    )
    insights['throughput_per_week'] = throughput.to_dict()

    # --- 3. User utilization ---
    # Sum hours logged per user over period, then divide by working_hours
    user_hours = (
        tl.groupby('user_id')['cycle_time_h']
          .sum()
    )
    insights['user_utilization'] = {
        user: round(hours / working_hours_per_week, 2)
        for user, hours in user_hours.items()
    }

    # --- 4. Bottleneck identification ---
    # Top 3 slowest task types
    sorted_types = insights['avg_cycle_time_by_type_h']
    insights['top_bottleneck_types'] = sorted(
        sorted_types.items(), key=lambda kv: kv[1], reverse=True
    )[:3]

    # Overdue tasks (created but not completed and > SLA threshold)
    now = pd.Timestamp.now()
    task_logs['created_at']   = pd.to_datetime(task_logs['created_at'])
    task_logs['completed_at'] = pd.to_datetime(task_logs['completed_at'])
    sla_hours = 48  # example SLA
    open_tasks = task_logs[task_logs['completed_at'].isna()].copy()
    open_tasks['age_h'] = (now - open_tasks['created_at']).dt.total_seconds() / 3600.0
    insights['overdue_tasks'] = open_tasks[open_tasks['age_h'] > sla_hours]['task_id'].tolist()

    # --- 5. Task aging distribution ---
    bins = [0, 24, 48, 72, 168, float('inf')]  # in hours: 0–1d,1–2d,2–3d,3–7d,>7d
    labels = ['<1d','1–2d','2–3d','3–7d','>7d']
    open_tasks['age_bucket'] = pd.cut(open_tasks['age_h'], bins=bins, labels=labels)
    insights['open_task_aging'] = (
        open_tasks['age_bucket']
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return insights
