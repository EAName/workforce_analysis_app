# AI Workforce Analysis App

A comprehensive Streamlit application for analyzing workforce data using AI-powered agents to provide insights on attrition, diversity, skills, and planning.

## Features

### Core Features
- Interactive data visualization and analysis
- AI-powered workforce insights
- Data validation and preprocessing
- Export capabilities for reports and recommendations
- Comprehensive test suite

### Analysis Capabilities
1. **Attrition Analysis**
   - Predict employee attrition risk
   - Identify high-risk employees
   - Analyze feature importance
   - Generate retention recommendations

2. **Diversity Analysis**
   - Gender ratio and distribution
   - Education field and level distribution
   - Leadership diversity metrics
   - Pay equity analysis
   - Department distribution
   - Turnover analysis by gender

3. **Skill Gap Analysis**
   - Role-based skill requirements
   - Resume and transcript analysis
   - Missing skills identification
   - Training recommendations
   - Custom course mapping

4. **Workforce Planning**
   - Headcount forecasting
   - Budget impact analysis
   - Role-based hiring plans
   - Cost projections

5. **Simulation**
   - Attrition intervention simulation
   - Cost-benefit analysis
   - Participation rate modeling
   - Retention impact projection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EAName/workforce_analysis_app.git
cd workforce_analysis_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your HR data in CSV format (see Data Requirements below)

4. Navigate through the analysis tabs to explore different insights

## Data Requirements

The application expects HR data in CSV format with the following required columns:

### Required Columns
- `EmployeeNumber`: Unique identifier for each employee
- `Age`: Employee age
- `Department`: Employee department
- `JobRole`: Employee job role
- `Salary`: Annual salary
- `YearsAtCompany`: Years of employment
- `JobSatisfaction`: Job satisfaction score (1-5)
- `WorkLifeBalance`: Work-life balance score (1-5)
- `PerformanceRating`: Performance rating (1-5)
- `Attrition`: Whether the employee left the company (Yes/No)
- `HireDate`: Employee hire date
- `Education`: Education level (1-5)
- `EducationField`: Field of education
- `Gender`: Employee gender
- `MaritalStatus`: Employee marital status

### Optional Columns
- `TerminationDate`: Employee termination date
- `NumCompaniesWorked`: Number of companies worked at
- `TotalWorkingYears`: Total years of work experience
- `TrainingTimesLastYear`: Number of training sessions attended
- `YearsInCurrentRole`: Years in current position
- `YearsSinceLastPromotion`: Years since last promotion
- `YearsWithCurrManager`: Years with current manager

## Project Structure

```
workforce_analysis_app/
├── app.py                 # Main Streamlit application
├── config/               # Configuration files
│   ├── config.py        # Configuration management
│   └── default_config.yaml
├── agents/              # AI analysis agents
│   ├── base_agent.py    # Base agent class
│   ├── attrition_agent.py    # Attrition prediction
│   ├── diversity_agent.py    # Diversity metrics
│   ├── skill_gap_agent.py    # Skill analysis
│   ├── planning_agent.py     # Workforce planning
│   └── simulation_agent.py   # Intervention simulation
├── schemas/             # Data validation schemas
│   └── data_schema.py
├── utils/              # Utility functions
│   ├── data_loader.py
│   └── logger.py
├── tests/              # Test suite
│   ├── test_attrition_agent.py
│   ├── test_diversity_agent.py
│   ├── test_skill_gap_agent.py
│   ├── test_planning_agent.py
│   └── test_simulation_agent.py
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## AI Agents

### Base Agent
- Provides common functionality for all agents
- Handles data validation and preprocessing
- Manages logging and error handling

### Attrition Agent
- Predicts employee attrition risk using Random Forest
- Identifies high-risk employees
- Analyzes feature importance
- Provides retention recommendations

### Diversity Agent
- Calculates gender ratio and distribution
- Analyzes education and leadership diversity
- Monitors pay equity
- Tracks turnover by demographic

### Skill Gap Agent
- Maps required skills to job roles
- Analyzes employee skills from resumes
- Identifies skill gaps
- Recommends training programs

### Planning Agent
- Forecasts hiring needs
- Analyzes budget impact
- Generates role-based hiring plans
- Projects costs and timelines

### Simulation Agent
- Simulates retention interventions
- Models participation rates
- Projects cost-benefit ratios
- Analyzes intervention effectiveness

## Configuration

The application uses a YAML-based configuration system. You can modify settings in `config/default_config.yaml` or override them using environment variables.

### Key Configuration Options
- Model parameters
- Data validation rules
- Logging settings
- UI customization
- Feature thresholds

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## API Documentation

### Base Agent
```python
class BaseAgent:
    def __init__(self, agent_name: str):
        """
        Initialize the base agent.
        
        Args:
            agent_name (str): Name of the agent for logging and identification
        """
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        """
        Validate input data against schema.
        
        Args:
            df (pd.DataFrame): Input data to validate
            
        Returns:
            bool: True if validation passes, raises ValueError otherwise
        """
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data.
        
        Args:
            df (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
```

### Attrition Agent
```python
class AttritionAgent(BaseAgent):
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze attrition risk for the given data.
        
        Args:
            df (pd.DataFrame): Employee data with required features
            
        Returns:
            Dict[str, Any]: Analysis results containing:
                - risk_scores: Array of attrition risk scores
                - high_risk_employees: DataFrame of high-risk employees
                - metrics: Dictionary of analysis metrics
                - feature_importance: Series of feature importance scores
        """
        
    def train_model(self, df: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler, List[str]]:
        """
        Train the attrition prediction model.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            Tuple containing:
                - Trained RandomForest model
                - Fitted StandardScaler
                - List of feature columns
        """
        
    def get_feature_importance(self) -> pd.Series:
        """
        Get feature importance from the trained model.
        
        Returns:
            pd.Series: Feature importance scores
        """
```

### Diversity Agent
```python
def monitor_diversity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Monitor diversity metrics from employee data.
    
    Args:
        df (pd.DataFrame): Employee data with required columns
        
    Returns:
        Dict[str, Any]: Diversity metrics including:
            - gender_ratio: Female to male ratio
            - education_field_distribution: Distribution of education fields
            - female_leadership_ratio: Ratio of female leaders
            - turnover_by_gender: Turnover distribution by gender
            - median_salary_by_gender: Median salary by gender
            - pay_equity_ratio: Female to male salary ratio
            - education_level_distribution: Distribution of education levels
            - marital_status_distribution: Distribution of marital status
            - department_distribution: Distribution across departments
    """
```

### Skill Gap Agent
```python
def analyze_skill_gap(
    df: pd.DataFrame,
    resume_texts: Dict[int, str],
    transcripts: Dict[int, List[str]],
    skill_course_map: Dict[str, str]
) -> List[Dict[str, Any]]:
    """
    Analyze skill gaps and recommend trainings.
    
    Args:
        df (pd.DataFrame): Employee data with columns:
            - EmployeeNumber: Unique ID
            - JobRole: Job role
        resume_texts (Dict[int, str]): Raw resume text per employee_id
        transcripts (Dict[int, List[str]]): List of completed training titles per employee_id
        skill_course_map (Dict[str, str]): Maps missing skills to recommended courses
        
    Returns:
        List[Dict[str, Any]]: List of recommendations, each containing:
            - employee_id: Employee identifier
            - job_role: Current job role
            - missing_skills: List of missing skills
            - recommendations: List of recommended courses
    """
```

### Planning Agent
```python
def forecast_workforce_plan(
    headcount_plan: pd.DataFrame,
    hiring_pipeline: pd.DataFrame
) -> Dict[str, Any]:
    """
    Forecast hiring needs and budget impact.
    
    Args:
        headcount_plan (pd.DataFrame): Plan with columns:
            - role: Job role
            - planned_hires: Number of planned hires
            - avg_salary: Average salary for the role
        hiring_pipeline (pd.DataFrame): Pipeline data with columns:
            - role: Job role
            - conversion_rate: Expected conversion rate
            
    Returns:
        Dict[str, Any]: Forecast results including:
            - next_quarter_hires: Expected number of hires
            - budget_impact: Total budget impact
            - by_role: List of role-specific forecasts
    """
```

### Simulation Agent
```python
def simulate_attrition_interventions(
    df: pd.DataFrame,
    intervention: Dict[str, Any],
    participation_rate: float = 1.0
) -> Dict[str, Any]:
    """
    Simulate retention scenarios by reducing attrition risk.
    
    Args:
        df (pd.DataFrame): Employee data with columns:
            - EmployeeNumber: Unique ID
            - Attrition: Attrition status
        intervention (Dict[str, Any]): Intervention details:
            - type: Intervention type
            - effect_size_pct: Expected effect size
            - cost_per_employee: Cost per participant
        participation_rate (float): Expected participation rate (0.0-1.0)
        
    Returns:
        Dict[str, Any]: Simulation results including:
            - baseline_attrition_rate: Current attrition rate
            - projected_attrition_rate: Expected rate after intervention
            - baseline_retention_rate: Current retention rate
            - projected_retention_rate: Expected retention rate
            - employees_participating: Number of participants
            - attritions_rescued: Expected number of prevented attritions
            - intervention_cost: Total intervention cost
            - intervention_type: Type of intervention
    """
```

### Usage Examples

#### Attrition Analysis
```python
from agents.attrition_agent import AttritionAgent

# Initialize agent
agent = AttritionAgent()

# Run analysis
results = agent.analyze(employee_data)

# Access results
risk_scores = results['risk_scores']
high_risk_employees = results['high_risk_employees']
feature_importance = results['feature_importance']
```

#### Diversity Analysis
```python
from agents.diversity_agent import monitor_diversity

# Run analysis
metrics = monitor_diversity(employee_data)

# Access metrics
gender_ratio = metrics['gender_ratio']
pay_equity = metrics['pay_equity_ratio']
```

#### Skill Gap Analysis
```python
from agents.skill_gap_agent import analyze_skill_gap

# Run analysis
recommendations = analyze_skill_gap(
    employee_data,
    resume_texts,
    training_transcripts,
    skill_course_mapping
)

# Process recommendations
for rec in recommendations:
    print(f"Employee {rec['employee_id']} needs training in: {rec['missing_skills']}")
```

#### Workforce Planning
```python
from agents.planning_agent import forecast_workforce_plan

# Run forecast
forecast = forecast_workforce_plan(headcount_plan, hiring_pipeline)

# Access results
next_quarter_hires = forecast['next_quarter_hires']
budget_impact = forecast['budget_impact']
```

#### Simulation
```python
from agents.simulation_agent import simulate_attrition_interventions

# Define intervention
intervention = {
    'type': 'Career Development Program',
    'effect_size_pct': 30.0,
    'cost_per_employee': 5000.0
}

# Run simulation
results = simulate_attrition_interventions(
    employee_data,
    intervention,
    participation_rate=0.8
)

# Access results
projected_attrition = results['projected_attrition_rate']
total_cost = results['intervention_cost']
``` 