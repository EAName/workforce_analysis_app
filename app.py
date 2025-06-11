import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import data_loader
from utils.logger import logger
from config.config import config
from agents.attrition_agent import AttritionAgent
from agents.simulation_agent import simulate_attrition_interventions
from agents.skill_gap_agent import analyze_skill_gap
from agents.diversity_agent import monitor_diversity
from agents.planning_agent import forecast_workforce_plan
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'agent' not in st.session_state:
    st.session_state.agent = None

def main():
    st.set_page_config(
        page_title="AI Workforce Analysis",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("AI Workforce Analysis")
    st.markdown("""
    This application uses AI to analyze workforce data and provide insights on attrition, diversity, skills, and planning.
    Upload your HR data to get started.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload HR Data (CSV)",
            type=['csv'],
            help="Upload a CSV file containing employee data"
        )
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                temp_path = Path(config.paths.temp_dir) / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Load and process data
                df = data_loader.load_data(str(temp_path))
                st.session_state.data = df
                
                # Clean up temporary file
                os.remove(temp_path)
                
                st.success("Data loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                logger.error(f"Error loading data: {str(e)}")
                return
    
    # Main content
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Data Overview
        st.header("Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Employees", len(df))
        with col2:
            st.metric("Departments", df['Department'].nunique())
        with col3:
            st.metric("Job Roles", df['JobRole'].nunique())
        
        # Data Visualization
        st.header("Data Visualization")
        
        # Department Distribution
        fig_dept = px.pie(
            df,
            names='Department',
            title='Employee Distribution by Department'
        )
        st.plotly_chart(fig_dept, use_container_width=True)
        
        # Age Distribution
        fig_age = px.histogram(
            df,
            x='Age',
            nbins=30,
            title='Age Distribution'
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        # Salary Distribution by Department
        fig_salary = px.box(
            df,
            x='Department',
            y='Salary',
            title='Salary Distribution by Department'
        )
        st.plotly_chart(fig_salary, use_container_width=True)
        
        # Analysis Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Attrition Analysis", 
            "Diversity Analysis",
            "Skill Gap Analysis",
            "Workforce Planning",
            "Simulation"
        ])
        
        with tab1:
            st.header("Attrition Analysis")
            if st.button("Run Attrition Analysis"):
                try:
                    with st.spinner("Running analysis..."):
                        # Initialize agent if not exists
                        if st.session_state.agent is None:
                            st.session_state.agent = AttritionAgent()
                        
                        # Run analysis
                        results = st.session_state.agent.analyze(df)
                        
                        # Display results
                        st.subheader("Attrition Risk Factors")
                        
                        # Feature importance plot
                        fig_importance = px.bar(
                            x=results['feature_importance'].index,
                            y=results['feature_importance'].values,
                            title='Feature Importance for Attrition Prediction'
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
                        
                        # Risk distribution
                        fig_risk = px.histogram(
                            x=results['risk_scores'],
                            nbins=30,
                            title='Distribution of Attrition Risk Scores'
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)
                        
                        # High-risk employees
                        st.subheader("High-Risk Employees")
                        high_risk = results['high_risk_employees']
                        st.dataframe(high_risk)
                        
                        # Download results
                        csv = high_risk.to_csv(index=False)
                        st.download_button(
                            "Download High-Risk Employees",
                            csv,
                            "high_risk_employees.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    logger.error(f"Error running analysis: {str(e)}")
        
        with tab2:
            st.header("Diversity Analysis")
            if st.button("Run Diversity Analysis"):
                try:
                    with st.spinner("Running diversity analysis..."):
                        # Run diversity analysis
                        diversity_metrics = monitor_diversity(df)
                        
                        # Display metrics
                        st.subheader("Diversity Metrics")
                        
                        # Gender ratio
                        st.metric(
                            "Gender Ratio (Female/Male)",
                            f"{diversity_metrics['gender_ratio']:.2%}"
                        )
                        
                        # Education field distribution
                        st.subheader("Education Field Distribution")
                        fig_education = px.pie(
                            values=list(diversity_metrics['education_field_distribution'].values()),
                            names=list(diversity_metrics['education_field_distribution'].keys()),
                            title='Education Field Distribution'
                        )
                        st.plotly_chart(fig_education, use_container_width=True)
                        
                        # Leadership diversity
                        st.metric(
                            "Female Leadership Ratio",
                            f"{diversity_metrics['female_leadership_ratio']:.2%}"
                        )
                        
                        # Education level distribution
                        st.subheader("Education Level Distribution")
                        fig_edu_level = px.pie(
                            values=list(diversity_metrics['education_level_distribution'].values()),
                            names=list(diversity_metrics['education_level_distribution'].keys()),
                            title='Education Level Distribution'
                        )
                        st.plotly_chart(fig_edu_level, use_container_width=True)
                        
                        # Marital status distribution
                        st.subheader("Marital Status Distribution")
                        fig_marital = px.pie(
                            values=list(diversity_metrics['marital_status_distribution'].values()),
                            names=list(diversity_metrics['marital_status_distribution'].keys()),
                            title='Marital Status Distribution'
                        )
                        st.plotly_chart(fig_marital, use_container_width=True)
                        
                        # Department distribution
                        st.subheader("Department Distribution")
                        fig_dept = px.pie(
                            values=list(diversity_metrics['department_distribution'].values()),
                            names=list(diversity_metrics['department_distribution'].keys()),
                            title='Department Distribution'
                        )
                        st.plotly_chart(fig_dept, use_container_width=True)
                        
                        # Pay equity
                        if diversity_metrics['pay_equity_ratio'] is not None:
                            st.metric(
                                "Pay Equity Ratio (Female/Male)",
                                f"{diversity_metrics['pay_equity_ratio']:.2%}"
                            )
                        
                        # Turnover by gender
                        st.subheader("Turnover by Gender")
                        fig_turnover = px.pie(
                            values=list(diversity_metrics['turnover_by_gender'].values()),
                            names=list(diversity_metrics['turnover_by_gender'].keys()),
                            title='Turnover Distribution by Gender'
                        )
                        st.plotly_chart(fig_turnover, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error running diversity analysis: {str(e)}")
                    logger.error(f"Error running diversity analysis: {str(e)}")
        
        with tab3:
            st.header("Skill Gap Analysis")
            if st.button("Run Skill Gap Analysis"):
                try:
                    with st.spinner("Running skill gap analysis..."):
                        # Prepare sample data for demonstration
                        resume_texts = {i: "Sample resume text" for i in df['EmployeeNumber']}
                        transcripts = {i: ["Sample training"] for i in df['EmployeeNumber']}
                        skill_course_map = {
                            "Python": "Python Programming Course",
                            "Machine Learning": "ML Fundamentals",
                            "Data Analysis": "Data Analysis Bootcamp"
                        }
                        
                        # Run skill gap analysis
                        skill_gaps = analyze_skill_gap(
                            df,
                            resume_texts,
                            transcripts,
                            skill_course_map
                        )
                        
                        # Display results
                        st.subheader("Skill Gap Analysis Results")
                        
                        # Convert to DataFrame for better display
                        gaps_df = pd.DataFrame(skill_gaps)
                        st.dataframe(gaps_df)
                        
                        # Download results
                        csv = gaps_df.to_csv(index=False)
                        st.download_button(
                            "Download Skill Gaps",
                            csv,
                            "skill_gaps.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error running skill gap analysis: {str(e)}")
                    logger.error(f"Error running skill gap analysis: {str(e)}")
        
        with tab4:
            st.header("Workforce Planning")
            if st.button("Run Workforce Planning"):
                try:
                    with st.spinner("Running workforce planning..."):
                        # Prepare sample data for demonstration
                        headcount_plan = pd.DataFrame({
                            'role': df['JobRole'].unique(),
                            'planned_hires': np.random.randint(1, 10, size=len(df['JobRole'].unique())),
                            'avg_salary': df.groupby('JobRole')['Salary'].mean().values
                        })
                        
                        hiring_pipeline = pd.DataFrame({
                            'role': df['JobRole'].unique(),
                            'conversion_rate': np.random.uniform(0.5, 0.9, size=len(df['JobRole'].unique()))
                        })
                        
                        # Run workforce planning
                        forecast = forecast_workforce_plan(headcount_plan, hiring_pipeline)
                        
                        # Display results
                        st.subheader("Workforce Planning Forecast")
                        
                        # Key metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Next Quarter Hires",
                                forecast['next_quarter_hires']
                            )
                        with col2:
                            st.metric(
                                "Budget Impact",
                                f"${forecast['budget_impact']:,.2f}"
                            )
                        
                        # Detailed breakdown
                        st.subheader("Hiring Plan by Role")
                        plan_df = pd.DataFrame(forecast['by_role'])
                        st.dataframe(plan_df)
                        
                        # Download results
                        csv = plan_df.to_csv(index=False)
                        st.download_button(
                            "Download Hiring Plan",
                            csv,
                            "hiring_plan.csv",
                            "text/csv"
                        )
                
                except Exception as e:
                    st.error(f"Error running workforce planning: {str(e)}")
                    logger.error(f"Error running workforce planning: {str(e)}")
        
        with tab5:
            st.header("Attrition Simulation")
            if st.button("Run Attrition Simulation"):
                try:
                    with st.spinner("Running attrition simulation..."):
                        # Define intervention
                        intervention = {
                            'type': 'Career Development Program',
                            'effect_size_pct': 30.0,  # 30% reduction in attrition risk
                            'cost_per_employee': 5000.0  # $5,000 per employee
                        }
                        
                        # Run simulation
                        simulation_results = simulate_attrition_interventions(
                            df,
                            intervention,
                            participation_rate=0.5  # 50% participation
                        )
                        
                        # Display results
                        st.subheader("Simulation Results")
                        
                        # Key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Baseline Attrition Rate",
                                f"{simulation_results['baseline_attrition_rate']:.1%}"
                            )
                        with col2:
                            st.metric(
                                "Projected Attrition Rate",
                                f"{simulation_results['projected_attrition_rate']:.1%}"
                            )
                        with col3:
                            st.metric(
                                "Attritions Rescued",
                                simulation_results['attritions_rescued']
                            )
                        
                        # Additional metrics
                        st.metric(
                            "Intervention Cost",
                            f"${simulation_results['intervention_cost']:,.2f}"
                        )
                        st.metric(
                            "Employees Participating",
                            simulation_results['employees_participating']
                        )
                
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
                    logger.error(f"Error running simulation: {str(e)}")

if __name__ == "__main__":
    main()
