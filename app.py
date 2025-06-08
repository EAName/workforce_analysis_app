import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import data_loader
from utils.logger import logger
from config.config import config
from agents.attrition_agent import AttritionAgent
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
    This application uses AI to analyze workforce data and predict employee attrition.
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
        
        # Attrition Analysis
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

if __name__ == "__main__":
    main()
