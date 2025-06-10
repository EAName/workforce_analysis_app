import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from utils.logger import logger
from config.config import config
from schemas.data_schema import HR_SCHEMA
import os
from pathlib import Path

class DataLoader:
    """Class for loading and validating HR data"""
    
    def __init__(self):
        self.config = config
        self.schema = HR_SCHEMA
        self.logger = logger
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file and validate against schema"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Check file size
            if os.path.getsize(file_path) > self.config.app.max_upload_size:
                raise ValueError(
                    f"File size exceeds maximum allowed size of "
                    f"{self.config.app.max_upload_size / (1024 * 1024)}MB"
                )
            
            # Read CSV file
            df = pd.read_csv(file_path, parse_dates=['HireDate', 'TerminationDate'])
            self.logger.info(f"Successfully loaded data with {len(df)} rows")
            
            # Validate data
            self._validate_data(df)
            
            # Preprocess data
            df = self._preprocess_data(df)
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data against schema"""
        try:
            # Validate against schema
            self.schema.validate_dataframe(df)
            
            # Check for missing values
            missing_ratio = df.isnull().sum() / len(df)
            high_missing_cols = missing_ratio[missing_ratio > self.config.data.max_missing_values]

            # Handle missing values in TerminationDate
            if 'TerminationDate' in high_missing_cols:
                df['TerminationDate'].fillna(pd.NaT, inplace=True)
                high_missing_cols = high_missing_cols.drop('TerminationDate')

            if not high_missing_cols.empty:
                if self.config.data.validation_strict:
                    raise ValueError(
                        f"Columns with too many missing values: {high_missing_cols.index.tolist()}"
                    )
                else:
                    self.logger.warning(
                        f"Columns with high missing values: {high_missing_cols.index.tolist()}"
                    )
            
            return True
        
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for analysis."""
        # Create a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Convert categorical variables to dummy variables
        categorical_cols = [col for col in self.schema.categorical_columns 
                          if col not in ['Department', 'JobRole']]  # Exclude Department and JobRole
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
        
        # Handle missing values
        numeric_cols = self.schema.numeric_columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """Save processed data to file"""
        try:
            output_path = Path(self.config.paths.data_dir) / filename
            df.to_csv(output_path, index=False)
            self.logger.info(f"Processed data saved to {output_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            raise

# Create singleton instance
data_loader = DataLoader()
