from typing import Dict, Any, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from agents.skill_gap_agent import analyze_skill_gap
from api.base_api import BaseAPI

class SkillGapRequest(BaseModel):
    """Request model for skill gap analysis"""
    data: Dict[str, List[Any]]  # DataFrame as dictionary
    resume_texts: Dict[str, str]  # Employee ID to resume text mapping
    transcripts: Dict[str, str]  # Employee ID to training transcript mapping

class SkillGapResponse(BaseModel):
    """Response model for skill gap analysis"""
    recommendations: List[Dict[str, Any]]
    missing_skills: Dict[str, List[str]]
    training_recommendations: Dict[str, List[str]]

class SkillGapAPI(BaseAPI):
    """API for skill gap analysis"""
    
    def __init__(self, app: FastAPI):
        """Initialize the skill gap API"""
        super().__init__(app, None)  # No agent instance needed for this API
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/skill-gap/analyze", response_model=SkillGapResponse)
        async def analyze_skill_gaps(request: SkillGapRequest):
            """
            Analyze skill gaps for the given data.
            
            Args:
                request (SkillGapRequest): Request containing employee data and documents
                
            Returns:
                SkillGapResponse: Analysis results
            """
            try:
                # Convert request data to DataFrame
                df = pd.DataFrame(request.data)
                
                # Run analysis
                results = analyze_skill_gap(
                    df,
                    request.resume_texts,
                    request.transcripts
                )
                
                # Process results into response format
                recommendations = []
                missing_skills = {}
                training_recommendations = {}
                
                for result in results:
                    employee_id = result['employee_id']
                    recommendations.append({
                        'employee_id': employee_id,
                        'missing_skills': result['missing_skills'],
                        'recommendations': result['recommendations']
                    })
                    missing_skills[employee_id] = result['missing_skills']
                    training_recommendations[employee_id] = result['recommendations']
                
                return SkillGapResponse(
                    recommendations=recommendations,
                    missing_skills=missing_skills,
                    training_recommendations=training_recommendations
                )
                
            except Exception as e:
                self.handle_error(e)
        
        @self.app.get("/api/skill-gap/required-skills")
        async def get_required_skills():
            """
            Get required skills for different roles.
            
            Returns:
                Dict[str, List[str]]: Role to required skills mapping
            """
            try:
                # This would typically come from a database or configuration
                return {
                    "Software Engineer": ["Python", "SQL", "Git", "Docker"],
                    "Data Scientist": ["Python", "R", "SQL", "Machine Learning"],
                    "Product Manager": ["Agile", "JIRA", "Product Strategy"],
                    "UX Designer": ["Figma", "User Research", "Prototyping"]
                }
            except Exception as e:
                self.handle_error(e) 