import re
from collections import defaultdict

def analyze_skill_gap(df, resume_texts, transcripts, skill_course_map):
    """
    Analyze skill gaps and recommend trainings.

    Parameters:
    - df: DataFrame with columns:
         'employee_id'         – unique ID
         'role'                – job role
         'required_skills'     – list of skills required for that role
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
        emp_id = row['employee_id']
        required = {s.lower() for s in row['required_skills']}
        
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
            'missing_skills': missing,
            'recommendations': recs or ["No mapped course; consider custom training"]
        })
    
    return recommendations
