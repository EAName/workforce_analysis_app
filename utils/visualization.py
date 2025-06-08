import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def create_attrition_risk_plot(df: pd.DataFrame) -> go.Figure:
    """Create an interactive plot showing attrition risk distribution"""
    fig = px.histogram(
        df,
        x='attrition_risk',
        nbins=20,
        title='Distribution of Attrition Risk Scores',
        labels={'attrition_risk': 'Attrition Risk Score', 'count': 'Number of Employees'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title='Attrition Risk Score',
        yaxis_title='Number of Employees'
    )
    return fig

def create_department_attrition_plot(df: pd.DataFrame) -> go.Figure:
    """Create a plot showing attrition by department"""
    dept_attrition = df.groupby('Department')['attrition_risk'].mean().reset_index()
    fig = px.bar(
        dept_attrition,
        x='Department',
        y='attrition_risk',
        title='Average Attrition Risk by Department',
        labels={'attrition_risk': 'Average Attrition Risk', 'Department': 'Department'},
        color='attrition_risk',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_layout(
        xaxis_title='Department',
        yaxis_title='Average Attrition Risk',
        coloraxis_showscale=False
    )
    return fig

def create_salary_attrition_scatter(df: pd.DataFrame) -> go.Figure:
    """Create a scatter plot of salary vs attrition risk"""
    fig = px.scatter(
        df,
        x='Salary',
        y='attrition_risk',
        color='Department',
        title='Salary vs Attrition Risk by Department',
        labels={
            'Salary': 'Annual Salary ($)',
            'attrition_risk': 'Attrition Risk Score',
            'Department': 'Department'
        },
        hover_data=['EmployeeNumber', 'JobRole']
    )
    fig.update_layout(
        xaxis_title='Annual Salary ($)',
        yaxis_title='Attrition Risk Score'
    )
    return fig

def create_attrition_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a heatmap of attrition risk factors"""
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Correlation Heatmap of Attrition Risk Factors',
        color_continuous_scale='RdYlBu_r',
        aspect='auto'
    )
    fig.update_layout(
        xaxis_title='Features',
        yaxis_title='Features'
    )
    return fig

def create_attrition_trend_plot(df: pd.DataFrame) -> go.Figure:
    """Create a plot showing attrition trends over time"""
    # Assuming there's a 'Date' column, if not, you'll need to modify this
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        monthly_attrition = df.groupby(df['Date'].dt.to_period('M'))['attrition_risk'].mean()
        
        fig = px.line(
            x=monthly_attrition.index.astype(str),
            y=monthly_attrition.values,
            title='Monthly Attrition Risk Trend',
            labels={'x': 'Month', 'y': 'Average Attrition Risk'}
        )
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Average Attrition Risk'
        )
        return fig
    return None

def create_dashboard_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Create summary metrics for the dashboard"""
    metrics = {
        'total_employees': len(df),
        'high_risk_count': len(df[df['attrition_risk'] > 0.7]),
        'avg_attrition_risk': df['attrition_risk'].mean(),
        'departments_at_risk': df.groupby('Department')['attrition_risk']
            .mean()
            .sort_values(ascending=False)
            .head(3)
            .to_dict()
    }
    return metrics 