from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from agents.base_agent import BaseAgent

class BaseAPI:
    """Base API class for all agent APIs"""
    
    def __init__(self, app: FastAPI, agent: BaseAgent):
        """
        Initialize the API.
        
        Args:
            app (FastAPI): FastAPI application instance
            agent (BaseAgent): Agent instance to use
        """
        self.app = app
        self.agent = agent
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes. Override in child classes."""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data.
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            HTTPException: If validation fails
        """
        try:
            return self.agent.validate_input(data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """
        Handle API errors.
        
        Args:
            error (Exception): Error to handle
            
        Returns:
            Dict[str, Any]: Error response
        """
        if isinstance(error, ValueError):
            raise HTTPException(status_code=400, detail=str(error))
        elif isinstance(error, KeyError):
            raise HTTPException(status_code=400, detail=f"Missing required field: {str(error)}")
        else:
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(error)}") 