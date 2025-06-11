from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from agents.simulation_agent import simulate_attrition_interventions
from api.base_api import BaseAPI

class SimulationRequest(BaseModel):
    """Request model for attrition simulation"""
    data: Dict[str, List[Any]]  # DataFrame as dictionary
    intervention_impact: float  # Expected impact of intervention (0-1)
    time_horizon: int  # Number of months to simulate

class SimulationResponse(BaseModel):
    """Response model for attrition simulation"""
    baseline_attrition: float
    projected_attrition: float
    improvement: float
    monthly_projections: List[Dict[str, Any]]
    cost_savings: float

class SimulationAPI(BaseAPI):
    """API for attrition simulation"""
    
    def __init__(self, app: FastAPI):
        """Initialize the simulation API"""
        super().__init__(app, None)  # No agent instance needed for this API
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/simulation/attrition", response_model=SimulationResponse)
        async def simulate_attrition(request: SimulationRequest):
            """
            Simulate attrition scenarios with interventions.
            
            Args:
                request (SimulationRequest): Request containing employee data and simulation parameters
                
            Returns:
                SimulationResponse: Simulation results
            """
            try:
                # Convert request data to DataFrame
                df = pd.DataFrame(request.data)
                
                # Run simulation
                results = simulate_attrition_interventions(
                    df,
                    request.intervention_impact,
                    request.time_horizon
                )
                
                # Convert results to response format
                return SimulationResponse(
                    baseline_attrition=results['baseline_attrition'],
                    projected_attrition=results['projected_attrition'],
                    improvement=results['improvement'],
                    monthly_projections=results['monthly_projections'],
                    cost_savings=results['cost_savings']
                )
                
            except Exception as e:
                self.handle_error(e)
        
        @self.app.get("/api/simulation/interventions")
        async def get_interventions():
            """
            Get list of available interventions and their typical impact ranges.
            
            Returns:
                Dict[str, Dict[str, float]]: Intervention to impact range mapping
            """
            try:
                # This would typically come from a database or configuration
                return {
                    "Career Development Program": {
                        "min_impact": 0.1,
                        "max_impact": 0.3,
                        "typical_impact": 0.2
                    },
                    "Flexible Work Arrangements": {
                        "min_impact": 0.05,
                        "max_impact": 0.15,
                        "typical_impact": 0.1
                    },
                    "Compensation Adjustment": {
                        "min_impact": 0.15,
                        "max_impact": 0.25,
                        "typical_impact": 0.2
                    },
                    "Mentorship Program": {
                        "min_impact": 0.08,
                        "max_impact": 0.18,
                        "typical_impact": 0.13
                    }
                }
            except Exception as e:
                self.handle_error(e) 