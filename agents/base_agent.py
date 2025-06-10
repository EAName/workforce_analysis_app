from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, List
from utils.logger import logger
from config.config import config
from sklearn.preprocessing import StandardScaler, LabelEncoder
from schemas.data_schema import validate_dataframe

class BaseAgent(ABC):
    """Base class for all analysis agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger
        self.config = config
        self.scalers = {}
        self.encoders = {}
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform analysis on the data"""
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """Validate input data using schema validation"""
        try:
            # Check if data is empty
            if data.empty:
                self.logger.error("Input data is empty")
                return False
            
            # Use schema validation
            try:
                validate_dataframe(data)
                return True
            except ValueError as e:
                self.logger.error(f"Schema validation failed: {str(e)}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error validating input: {str(e)}")
            return False
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data according to configuration"""
        try:
            df = data.copy()
            
            # Handle missing values
            if 'handle_missing_values' in self.config.data.preprocessing_steps:
                for col in self.config.data.numeric_columns:
                    if col in df.columns and df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
                
                for col in self.config.data.categorical_columns:
                    if col in df.columns and df[col].isnull().any():
                        df[col].fillna(df[col].mode()[0], inplace=True)
            
            # Convert dates
            if 'convert_dates' in self.config.data.preprocessing_steps:
                for col in self.config.data.date_columns:
                    if col in df.columns:
                        try:
                            df[col] = pd.to_datetime(df[col])
                        except:
                            self.logger.warning(f"Could not convert {col} to datetime")
            
            # Encode categorical variables
            if 'encode_categorical' in self.config.data.preprocessing_steps:
                for col in self.config.data.categorical_columns:
                    if col in df.columns:
                        if col not in self.encoders:
                            self.encoders[col] = LabelEncoder()
                            df[col] = self.encoders[col].fit_transform(df[col])
                        else:
                            df[col] = self.encoders[col].transform(df[col])
            
            # Scale numeric variables
            if 'scale_numeric' in self.config.data.preprocessing_steps:
                for col in self.config.data.numeric_columns:
                    if col in df.columns:
                        if col not in self.scalers:
                            self.scalers[col] = StandardScaler()
                            df[col] = self.scalers[col].fit_transform(df[col].values.reshape(-1, 1))
                        else:
                            df[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1))
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save analysis results"""
        try:
            import json
            from pathlib import Path
            
            # Create results directory if it doesn't exist
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Save results
            with open(results_dir / filename, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.logger.info(f"Results saved to {filename}")
        
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise
    
    def load_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load saved analysis results"""
        try:
            import json
            from pathlib import Path
            
            results_file = Path("results") / filename
            if not results_file.exists():
                self.logger.warning(f"Results file {filename} not found")
                return None
            
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error loading results: {str(e)}")
            return None 