from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.attrition_api import AttritionAPI
from api.diversity_api import DiversityAPI
from api.skill_gap_api import SkillGapAPI
from api.planning_api import PlanningAPI
from api.simulation_api import SimulationAPI

app = FastAPI(
    title="Workforce Analysis API",
    description="API for workforce analysis and planning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize and setup all agent APIs
attrition_api = AttritionAPI(app)
diversity_api = DiversityAPI(app)
skill_gap_api = SkillGapAPI(app)
planning_api = PlanningAPI(app)
simulation_api = SimulationAPI(app)

# Setup routes for each API
attrition_api.setup_routes()
diversity_api.setup_routes()
skill_gap_api.setup_routes()
planning_api.setup_routes()
simulation_api.setup_routes()

@app.get("/")
async def root():
    """Root endpoint returning API information"""
    return {
        "name": "Workforce Analysis API",
        "version": "1.0.0",
        "description": "API for workforce analysis and planning",
        "endpoints": {
            "attrition": "/api/attrition",
            "diversity": "/api/diversity",
            "skill_gap": "/api/skill-gap",
            "planning": "/api/planning",
            "simulation": "/api/simulation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 