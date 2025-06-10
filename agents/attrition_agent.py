import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent

class AttritionAgent(BaseAgent):
    """Agent for predicting employee attrition"""
    
    def __init__(self):
        super().__init__("attrition_agent")
        self.model = None
        self.scaler = None
        self.feature_columns = None
    
    def train_model(self, data: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler, list]:
        """Train the attrition prediction model"""
        try:
            # Preprocess data using the standalone function to drop date columns
            df = preprocess_data(data)
            
            # Prepare features and target
            X = df.drop(['Attrition', 'EmployeeNumber'], axis=1, errors='ignore')
            y = df['Attrition'].map({'Yes': 1, 'No': 0})
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.model.test_size,
                random_state=self.config.model.random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            rf_params = self.config.model.model_params["RandomForest"].copy() if self.config.model.model_params and "RandomForest" in self.config.model.model_params else {}
            rf_params.pop("n_estimators", None)
            rf_params.pop("random_state", None)
            model = RandomForestClassifier(
                n_estimators=self.config.model.n_estimators,
                random_state=self.config.model.random_state,
                **rf_params
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Log evaluation metrics
            self.logger.info("\nModel Evaluation:")
            self.logger.info(classification_report(y_test, y_pred))
            try:
                self.logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
            except ValueError as ve:
                self.logger.warning(f"ROC AUC Score could not be computed: {ve}")
            
            # Save model and scaler
            os.makedirs(self.config.model.model_dir, exist_ok=True)
            joblib.dump(model, os.path.join(self.config.model.model_dir, 'attrition_model.joblib'))
            joblib.dump(scaler, os.path.join(self.config.model.model_dir, 'attrition_scaler.joblib'))
            joblib.dump(X.columns.tolist(), os.path.join(self.config.model.model_dir, 'attrition_features.joblib'))
            
            return model, scaler, X.columns.tolist()
        
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
            
            # Load or train model if not already done
            if not hasattr(self, 'model') or not hasattr(self, 'feature_columns'):
                self.model, self.scaler, self.feature_columns = self.train_model(df)
            
            # Get predictions
            results = predict_attrition(df)
            
            # Calculate metrics
            high_risk_threshold = 0.7
            high_risk_count = len(results[results['attrition_risk'] > high_risk_threshold])
            avg_risk = results['attrition_risk'].mean()
            risk_distribution = results['attrition_risk'].describe()
            
            # Get feature importance
            feature_importance = self.get_feature_importance(df)
            
            return {
                'risk_scores': results,
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
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.Series:
        """Compute feature importance from the trained model"""
        if not hasattr(self, 'model') or not hasattr(self, 'feature_columns'):
            raise ValueError("Model not trained. Call train_model first.")
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
    df_processed = df_processed.fillna(df_processed.mean())
    
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
