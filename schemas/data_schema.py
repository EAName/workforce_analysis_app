from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
from enum import Enum
import pandas as pd

class ColumnType(str, Enum):
    """Enum for column data types"""
    INTEGER = "int64"
    FLOAT = "float64"
    STRING = "object"
    DATETIME = "datetime64[ns]"
    BOOLEAN = "bool"

class ColumnDefinition(BaseModel):
    """Definition of a data column"""
    name: str
    type: ColumnType
    required: bool = True
    description: Optional[str] = None
    allowed_values: Optional[List[Union[str, int, float]]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None

    @validator('allowed_values')
    def validate_allowed_values(cls, v, values):
        if v is not None and values.get('type') == ColumnType.INTEGER:
            if not all(isinstance(x, int) for x in v):
                raise ValueError("Allowed values for integer columns must be integers")
        return v

class DataSchema(BaseModel):
    """Schema for HR data"""
    columns: Dict[str, ColumnDefinition]
    primary_key: str = "EmployeeNumber"
    version: str = "1.0.0"

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate a DataFrame against the schema"""
        errors = []
        
        # Check required columns
        missing_cols = [col for col, defn in self.columns.items() 
                       if defn.required and col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # Check column types
        for col, defn in self.columns.items():
            if col in df.columns:
                expected_type = defn.type.value
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    errors.append(
                        f"Column {col} has incorrect type. "
                        f"Expected {expected_type}, got {actual_type}"
                    )

                # Check allowed values
                if defn.allowed_values is not None:
                    invalid_values = df[~df[col].isin(defn.allowed_values)][col].unique()
                    if len(invalid_values) > 0:
                        errors.append(
                            f"Column {col} contains invalid values: {invalid_values}"
                        )

                # Check numeric ranges
                if defn.type in [ColumnType.INTEGER, ColumnType.FLOAT]:
                    if defn.min_value is not None:
                        if (df[col] < defn.min_value).any():
                            errors.append(
                                f"Column {col} contains values below minimum {defn.min_value}"
                            )
                    if defn.max_value is not None:
                        if (df[col] > defn.max_value).any():
                            errors.append(
                                f"Column {col} contains values above maximum {defn.max_value}"
                            )

        if errors:
            raise ValueError("\n".join(errors))

        return True

# Define the HR data schema
HR_SCHEMA = DataSchema(
    columns={
        "EmployeeNumber": ColumnDefinition(
            name="EmployeeNumber",
            type=ColumnType.INTEGER,
            description="Unique identifier for each employee",
            min_value=1
        ),
        "Age": ColumnDefinition(
            name="Age",
            type=ColumnType.INTEGER,
            description="Employee age",
            min_value=18,
            max_value=100
        ),
        "Department": ColumnDefinition(
            name="Department",
            type=ColumnType.STRING,
            description="Employee department",
            allowed_values=["IT", "HR", "Finance", "Marketing", "Operations", "Sales", "Research", "Engineering"]
        ),
        "JobRole": ColumnDefinition(
            name="JobRole",
            type=ColumnType.STRING,
            description="Employee job role",
            allowed_values=[
                "Developer", "Engineer", "System Administrator", "IT Manager", "Technical Specialist",
                "HR Manager", "HR Specialist", "Recruiter", "HR Director",
                "Financial Analyst", "Accountant", "Finance Manager", "Controller",
                "Marketing Specialist", "Marketing Manager", "Brand Manager", "Marketing Director",
                "Operations Manager", "Operations Specialist", "Supply Chain Manager",
                "Sales Representative", "Sales Manager", "Account Executive", "Sales Director",
                "Research Scientist", "Research Analyst", "Research Director",
                "Senior Engineer", "Engineering Manager", "Technical Director"
            ]
        ),
        "Salary": ColumnDefinition(
            name="Salary",
            type=ColumnType.INTEGER,
            description="Annual salary",
            min_value=0
        ),
        "YearsAtCompany": ColumnDefinition(
            name="YearsAtCompany",
            type=ColumnType.INTEGER,
            description="Years of employment",
            min_value=0,
            max_value=50
        ),
        "JobSatisfaction": ColumnDefinition(
            name="JobSatisfaction",
            type=ColumnType.INTEGER,
            description="Job satisfaction score",
            min_value=1,
            max_value=5
        ),
        "WorkLifeBalance": ColumnDefinition(
            name="WorkLifeBalance",
            type=ColumnType.INTEGER,
            description="Work-life balance score",
            min_value=1,
            max_value=5
        ),
        "PerformanceRating": ColumnDefinition(
            name="PerformanceRating",
            type=ColumnType.INTEGER,
            description="Performance rating",
            min_value=1,
            max_value=5
        ),
        "Attrition": ColumnDefinition(
            name="Attrition",
            type=ColumnType.STRING,
            description="Whether the employee left the company",
            allowed_values=["Yes", "No"]
        ),
        "HireDate": ColumnDefinition(
            name="HireDate",
            type=ColumnType.DATETIME,
            description="Employee hire date"
        ),
        "TerminationDate": ColumnDefinition(
            name="TerminationDate",
            type=ColumnType.DATETIME,
            required=False,
            description="Employee termination date"
        ),
        "Education": ColumnDefinition(
            name="Education",
            type=ColumnType.INTEGER,
            description="Education level",
            min_value=1,
            max_value=5
        ),
        "EducationField": ColumnDefinition(
            name="EducationField",
            type=ColumnType.STRING,
            description="Field of education",
            allowed_values=["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"]
        ),
        "Gender": ColumnDefinition(
            name="Gender",
            type=ColumnType.STRING,
            description="Employee gender",
            allowed_values=["Male", "Female"]
        ),
        "MaritalStatus": ColumnDefinition(
            name="MaritalStatus",
            type=ColumnType.STRING,
            description="Employee marital status",
            allowed_values=["Single", "Married", "Divorced"]
        ),
        "NumCompaniesWorked": ColumnDefinition(
            name="NumCompaniesWorked",
            type=ColumnType.INTEGER,
            description="Number of companies worked at",
            min_value=0,
            max_value=20
        ),
        "TotalWorkingYears": ColumnDefinition(
            name="TotalWorkingYears",
            type=ColumnType.INTEGER,
            description="Total years of work experience",
            min_value=0,
            max_value=50
        ),
        "TrainingTimesLastYear": ColumnDefinition(
            name="TrainingTimesLastYear",
            type=ColumnType.INTEGER,
            description="Number of training sessions attended last year",
            min_value=0,
            max_value=10
        ),
        "YearsInCurrentRole": ColumnDefinition(
            name="YearsInCurrentRole",
            type=ColumnType.INTEGER,
            description="Years in current role",
            min_value=0,
            max_value=50
        ),
        "YearsSinceLastPromotion": ColumnDefinition(
            name="YearsSinceLastPromotion",
            type=ColumnType.INTEGER,
            description="Years since last promotion",
            min_value=0,
            max_value=50
        ),
        "YearsWithCurrManager": ColumnDefinition(
            name="YearsWithCurrManager",
            type=ColumnType.INTEGER,
            description="Years with current manager",
            min_value=0,
            max_value=50
        )
    }
)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """Validate a DataFrame against the HR schema"""
    return HR_SCHEMA.validate_dataframe(df) 