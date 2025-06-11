import re
from collections import defaultdict

# Define required skills for each job role
ROLE_SKILLS_MAP = {
    "Developer": ["Python", "SQL", "Git", "Agile", "Testing"],
    "Engineer": ["Python", "SQL", "System Design", "Problem Solving"],
    "System Administrator": ["Linux", "Networking", "Security", "Cloud"],
    "IT Manager": ["Project Management", "Leadership", "IT Strategy", "Budgeting"],
    "Technical Specialist": ["Technical Writing", "Problem Solving", "Documentation"],
    "HR Manager": ["Recruitment", "Employee Relations", "HR Policies", "Leadership"],
    "HR Specialist": ["Recruitment", "Employee Relations", "HR Policies"],
    "Recruiter": ["Talent Acquisition", "Interviewing", "ATS", "Communication"],
    "HR Director": ["Strategic Planning", "Leadership", "HR Strategy", "Change Management"],
    "Financial Analyst": ["Financial Modeling", "Excel", "Analysis", "Reporting"],
    "Accountant": ["Accounting", "Tax", "Financial Reporting", "Excel"],
    "Finance Manager": ["Financial Planning", "Leadership", "Budgeting", "Analysis"],
    "Controller": ["Accounting", "Financial Controls", "Compliance", "Leadership"],
    "Marketing Specialist": ["Digital Marketing", "Content Creation", "Analytics"],
    "Marketing Manager": ["Strategy", "Leadership", "Campaign Management"],
    "Brand Manager": ["Brand Strategy", "Marketing", "Communication"],
    "Marketing Director": ["Strategic Planning", "Leadership", "Brand Management"],
    "Operations Manager": ["Process Improvement", "Leadership", "Supply Chain"],
    "Operations Specialist": ["Process Improvement", "Data Analysis", "Documentation"],
    "Supply Chain Manager": ["Supply Chain", "Logistics", "Inventory Management"],
    "Sales Representative": ["Sales", "CRM", "Communication", "Negotiation"],
    "Sales Manager": ["Sales Strategy", "Leadership", "CRM", "Team Management"],
    "Account Executive": ["Sales", "Account Management", "Communication"],
    "Sales Director": ["Strategic Planning", "Leadership", "Sales Strategy"],
    "Research Scientist": ["Research", "Data Analysis", "Scientific Writing"],
    "Research Analyst": ["Research", "Data Analysis", "Reporting"],
    "Research Director": ["Research Strategy", "Leadership", "Project Management"],
    "Senior Engineer": ["System Design", "Leadership", "Technical Architecture"],
    "Engineering Manager": ["Technical Leadership", "Project Management", "Team Management"],
    "Technical Director": ["Technical Strategy", "Leadership", "Architecture"]
}

def analyze_skill_gap(df, resume_texts, transcripts, skill_course_map):
    """
    Analyze skill gaps and recommend trainings.

    Parameters:
    - df: DataFrame with columns:
         'EmployeeNumber'      – unique ID
         'JobRole'            – job role
    - resume_texts: dict[int → str]
        raw resume text per employee_id
    - transcripts: dict[int → list[str]]
        list of completed training titles or skill tokens per employee_id
    - skill_course_map: dict[str → str]
        maps a missing skill → recommended course name or link

    Returns:
    - recommendations: list of dicts, each with:
        'employee_id'
        'missing_skills'
        'recommendations'
    """
    recommendations = []
    
    # Precompile skill regex map for fast text search
    skill_patterns = {
        skill.lower(): re.compile(r'\b' + re.escape(skill.lower()) + r'\b')
        for skill in skill_course_map
    }
    
    for _, row in df.iterrows():
        emp_id = row['EmployeeNumber']
        job_role = row['JobRole']
        
        # Get required skills for the job role
        required = {s.lower() for s in ROLE_SKILLS_MAP.get(job_role, [])}
        
        # 1. Extract skills from resume
        text = resume_texts.get(emp_id, "").lower()
        found_from_resume = {
            skill for skill, pat in skill_patterns.items() if pat.search(text)
        }
        
        # 2. Extract skills from transcripts (already tokenized)
        found_from_transcript = {s.lower() for s in transcripts.get(emp_id, [])}
        
        # 3. Combine known skills
        known_skills = found_from_resume | found_from_transcript
        
        # 4. Identify gaps
        missing = sorted(required - known_skills)
        
        # 5. Map to course recommendations
        recs = [skill_course_map[skill] for skill in missing if skill in skill_course_map]
        
        recommendations.append({
            'employee_id': emp_id,
            'job_role': job_role,
            'missing_skills': missing,
            'recommendations': recs or ["No mapped course; consider custom training"]
        })
    
    return recommendations
