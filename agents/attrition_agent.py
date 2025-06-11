import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
import os
from typing import Dict, Any, Tuple, List
from agents.base_agent import BaseAgent

class AttritionAgent(BaseAgent):
    """Agent for predicting employee attrition"""
    
    def __init__(self):
        super().__init__("attrition_agent")
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def train_model(self, df: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler, List[str]]:
        """Train the attrition prediction model"""
        try:
            # Prepare features
            feature_columns = [
                'Age', 'Salary', 'YearsAtCompany', 'JobSatisfaction',
                'WorkLifeBalance', 'PerformanceRating', 'Education',
                'NumCompaniesWorked', 'TotalWorkingYears', 'TrainingTimesLastYear',
                'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
            ]
            
            # Create target variable (1 if left, 0 if stayed)
            df['attrition_target'] = (df['Attrition'] == 'Yes').astype(int)
            
            # Prepare features and target
            X = df[feature_columns]
            y = df['attrition_target']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            self.logger.info(f"Model accuracy: {accuracy:.2f}")
            
            return model, scaler, feature_columns
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def load_model(self) -> bool:
        """Load the trained model and scaler"""
        try:
            model_path = os.path.join(self.config.model.model_dir, 'attrition_model.joblib')
            scaler_path = os.path.join(self.config.model.model_dir, 'attrition_scaler.joblib')
            features_path = os.path.join(self.config.model.model_dir, 'attrition_features.joblib')
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
                return False
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(features_path)
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze attrition risk for the given data"""
        try:
            # Validate input data
            if not self.validate_input(df):
                raise ValueError("Invalid input data")
            
            # Ensure model is trained
            if not hasattr(self, 'model') or self.model is None:
                self.model, self.scaler, self.feature_columns = self.train_model(df)
                if self.model is None:
                    self.logger.error("Model training failed, model is None.")
                    raise ValueError("Model training failed, model is None.")
            
            # Get predictions
            results = predict_attrition(df)
            
            # If results is empty, raise a clear error
            if len(results) == 0:
                self.logger.error("Prediction results DataFrame is empty.")
                raise ValueError("Prediction results DataFrame is empty.")
            
            # Calculate metrics
            high_risk_threshold = 0.7
            high_risk_count = len(results[results['attrition_risk'] > high_risk_threshold])
            avg_risk = results['attrition_risk'].mean()
            risk_distribution = results['attrition_risk'].describe()
            
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            # Get high-risk employees
            high_risk_employees = df[results['attrition_risk'] > high_risk_threshold].copy()
            high_risk_employees['attrition_risk'] = results[results['attrition_risk'] > high_risk_threshold]['attrition_risk']
            
            return {
                'risk_scores': results['attrition_risk'].values,
                'high_risk_employees': high_risk_employees,
                'metrics': {
                    'high_risk_count': high_risk_count,
                    'avg_risk': avg_risk,
                    'risk_distribution': risk_distribution.to_dict()
                },
                'feature_importance': feature_importance
            }
        except Exception as e:
            self.logger.error(f"Error in analysis: {str(e)}")
            raise
    
    def get_feature_importance(self) -> pd.Series:
        """Compute feature importance from the trained model"""
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model is not trained. Cannot compute feature importance.")
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_columns).sort_values(ascending=False)

# Create singleton instance
attrition_agent = AttritionAgent()

def preprocess_data(df):
    """Preprocess the HR data for attrition prediction"""
    # Drop date columns and non-feature columns
    df_processed = df.drop(['HireDate', 'TerminationDate'], axis=1, errors='ignore')
    
    # Identify categorical columns except 'Attrition'
    categorical_cols = [col for col in df_processed.select_dtypes(include=['object']).columns if col != 'Attrition']
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    # Handle missing values
    df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
    
    return df_processed

def predict_attrition(df):
    """Compute attrition risk scores for each employee"""
    try:
        # Load saved model and scaler
        model = joblib.load('models/attrition_model.joblib')
        scaler = joblib.load('models/attrition_scaler.joblib')
        feature_columns = joblib.load('models/attrition_features.joblib')
    except:
        # If model doesn't exist, train new one using AttritionAgent
        agent = AttritionAgent()
        model, scaler, feature_columns = agent.train_model(df)
        joblib.dump(feature_columns, 'models/attrition_features.joblib')
    
    # Preprocess new data
    df_processed = preprocess_data(df)
    
    # Ensure all required features are present
    for col in feature_columns:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Select features in correct order
    X = df_processed[feature_columns]
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make predictions
    attrition_risk = model.predict_proba(X_scaled)[:, 1]
    
    # Create results dataframe
    results = pd.DataFrame({
        'EmployeeNumber': df['EmployeeNumber'],
        'attrition_risk': attrition_risk
    })
    
    return results
