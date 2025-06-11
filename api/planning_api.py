from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from agents.planning_agent import forecast_workforce_plan
from api.base_api import BaseAPI

class PlanningRequest(BaseModel):
    """Request model for workforce planning"""
    headcount_plan: Dict[str, List[Any]]  # DataFrame as dictionary
    hiring_pipeline: Dict[str, List[Any]]  # DataFrame as dictionary

class PlanningResponse(BaseModel):
    """Response model for workforce planning"""
    next_quarter_hires: int
    budget_impact: float
    by_role: List[Dict[str, Any]]

class PlanningAPI(BaseAPI):
    """API for workforce planning"""
    
    def __init__(self, app: FastAPI):
        """Initialize the planning API"""
        super().__init__(app, None)  # No agent instance needed for this API
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/planning/forecast", response_model=PlanningResponse)
        async def forecast_workforce(request: PlanningRequest):
            """
            Forecast workforce needs based on headcount plan and hiring pipeline.
            
            Args:
                request (PlanningRequest): Request containing headcount plan and hiring pipeline data
                
            Returns:
                PlanningResponse: Forecast results
            """
            try:
                # Convert request data to DataFrames
                headcount_plan = pd.DataFrame(request.headcount_plan)
                hiring_pipeline = pd.DataFrame(request.hiring_pipeline)
                
                # Run analysis
                results = forecast_workforce_plan(headcount_plan, hiring_pipeline)
                
                # Convert results to response format
                return PlanningResponse(
                    next_quarter_hires=results['next_quarter_hires'],
                    budget_impact=results['budget_impact'],
                    by_role=results['by_role']
                )
                
            except Exception as e:
                self.handle_error(e)
        
        @self.app.get("/api/planning/roles")
        async def get_roles():
            """
            Get list of available roles for planning.
            
            Returns:
                List[str]: List of role names
            """
            try:
                # This would typically come from a database or configuration
                return [
                    "Software Engineer",
                    "Data Scientist",
                    "Product Manager",
                    "UX Designer",
                    "DevOps Engineer",
                    "QA Engineer"
                ]
            except Exception as e:
                self.handle_error(e)
        
        @self.app.get("/api/planning/salary-ranges")
        async def get_salary_ranges():
            """
            Get salary ranges for different roles.
            
            Returns:
                Dict[str, Dict[str, float]]: Role to salary range mapping
            """
            try:
                # This would typically come from a database or configuration
                return {
                    "Software Engineer": {
                        "min": 80000,
                        "max": 150000,
                        "average": 115000
                    },
                    "Data Scientist": {
                        "min": 90000,
                        "max": 160000,
                        "average": 125000
                    },
                    "Product Manager": {
                        "min": 95000,
                        "max": 170000,
                        "average": 132500
                    }
                }
            except Exception as e:
                self.handle_error(e) 