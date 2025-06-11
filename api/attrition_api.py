from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from agents.attrition_agent import AttritionAgent
from api.base_api import BaseAPI

class AttritionRequest(BaseModel):
    """Request model for attrition analysis"""
    data: Dict[str, List[Any]]  # DataFrame as dictionary

class AttritionResponse(BaseModel):
    """Response model for attrition analysis"""
    risk_scores: List[float]
    high_risk_employees: Dict[str, List[Any]]
    metrics: Dict[str, Any]
    feature_importance: Dict[str, float]

class AttritionAPI(BaseAPI):
    """API for attrition analysis"""
    
    def __init__(self, app: FastAPI):
        """Initialize the attrition API"""
        super().__init__(app, AttritionAgent())
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/attrition/analyze", response_model=AttritionResponse)
        async def analyze_attrition(request: AttritionRequest):
            """
            Analyze attrition risk for the given data.
            
            Args:
                request (AttritionRequest): Request containing employee data
                
            Returns:
                AttritionResponse: Analysis results
            """
            try:
                # Convert request data to DataFrame
                df = pd.DataFrame(request.data)
                
                # Validate data
                self.validate_data(df)
                
                # Run analysis
                results = self.agent.analyze(df)
                
                # Convert results to response format
                return AttritionResponse(
                    risk_scores=results['risk_scores'].tolist(),
                    high_risk_employees=results['high_risk_employees'].to_dict(),
                    metrics=results['metrics'],
                    feature_importance=results['feature_importance'].to_dict()
                )
                
            except Exception as e:
                self.handle_error(e)
        
        @self.app.get("/api/attrition/feature-importance")
        async def get_feature_importance():
            """
            Get feature importance from the trained model.
            
            Returns:
                Dict[str, float]: Feature importance scores
            """
            try:
                importance = self.agent.get_feature_importance()
                return importance.to_dict()
            except Exception as e:
                self.handle_error(e) 