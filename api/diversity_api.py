from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from agents.diversity_agent import monitor_diversity
from api.base_api import BaseAPI

class DiversityRequest(BaseModel):
    """Request model for diversity analysis"""
    data: Dict[str, List[Any]]  # DataFrame as dictionary

class DiversityResponse(BaseModel):
    """Response model for diversity analysis"""
    gender_ratio: float
    ethnicity_distribution: Dict[str, float]
    female_leadership_ratio: float
    turnover_by_gender: Dict[str, float]
    median_salary_by_gender: Dict[str, float]
    pay_equity_ratio: float

class DiversityAPI(BaseAPI):
    """API for diversity analysis"""
    
    def __init__(self, app: FastAPI):
        """Initialize the diversity API"""
        super().__init__(app, None)  # No agent instance needed for this API
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/diversity/analyze", response_model=DiversityResponse)
        async def analyze_diversity(request: DiversityRequest):
            """
            Analyze diversity metrics for the given data.
            
            Args:
                request (DiversityRequest): Request containing employee data
                
            Returns:
                DiversityResponse: Analysis results
            """
            try:
                # Convert request data to DataFrame
                df = pd.DataFrame(request.data)
                
                # Run analysis
                results = monitor_diversity(df)
                
                # Convert results to response format
                return DiversityResponse(
                    gender_ratio=results['gender_ratio'],
                    ethnicity_distribution=results['ethnicity_distribution'],
                    female_leadership_ratio=results['female_leadership_ratio'],
                    turnover_by_gender=results['turnover_by_gender'],
                    median_salary_by_gender=results['median_salary_by_gender'],
                    pay_equity_ratio=results['pay_equity_ratio']
                )
                
            except Exception as e:
                self.handle_error(e)
        
        @self.app.get("/api/diversity/metrics")
        async def get_diversity_metrics():
            """
            Get current diversity metrics.
            
            Returns:
                Dict[str, Any]: Current diversity metrics
            """
            try:
                # This endpoint would typically fetch metrics from a database
                # For now, return a placeholder response
                return {
                    "message": "Diversity metrics endpoint - implement database integration"
                }
            except Exception as e:
                self.handle_error(e) 