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