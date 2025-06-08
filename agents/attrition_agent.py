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
            # Preprocess data
            df = self.preprocess_data(data)
            
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
            model = RandomForestClassifier(
                n_estimators=self.config.model.n_estimators,
                random_state=self.config.model.random_state,
                **self.config.model.model_params
            )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Log evaluation metrics
            self.logger.info("\nModel Evaluation:")
            self.logger.info(classification_report(y_test, y_pred))
            self.logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
            
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
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Analyze attrition risk for employees"""
        try:
            # Validate input
            if not self.validate_input(data):
                raise ValueError("Invalid input data")
            
            # Load or train model
            if not self.load_model():
                self.model, self.scaler, self.feature_columns = self.train_model(data)
            
            # Preprocess data
            df = self.preprocess_data(data)
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Select features in correct order
            X = df[self.feature_columns]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            attrition_risk = self.model.predict_proba(X_scaled)[:, 1]
            
            # Create results
            results = pd.DataFrame({
                'EmployeeNumber': data['EmployeeNumber'],
                'attrition_risk': attrition_risk
            })
            
            # Calculate additional metrics
            metrics = {
                'high_risk_count': len(results[results['attrition_risk'] > 0.7]),
                'avg_risk': results['attrition_risk'].mean(),
                'risk_distribution': results['attrition_risk'].describe().to_dict()
            }
            
            # Save results
            self.save_results({
                'predictions': results.to_dict(orient='records'),
                'metrics': metrics
            }, 'attrition_results.json')
            
            return {
                'predictions': results,
                'metrics': metrics
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing attrition: {str(e)}")
            raise

# Create singleton instance
attrition_agent = AttritionAgent()

def preprocess_data(df):
    """Preprocess the HR data for attrition prediction"""
    # Convert categorical variables to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
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
        # If model doesn't exist, train new one
        model, scaler, feature_columns = train_attrition_model(df)
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
