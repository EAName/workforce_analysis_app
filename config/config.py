from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from schemas.data_schema import HR_SCHEMA

@dataclass
class PathConfig:
    """Configuration for file paths"""
    base_dir: str = "."
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    results_dir: str = "results"
    config_dir: str = "config"
    temp_dir: str = "temp"

    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.data_dir, self.model_dir, self.log_dir, 
                        self.results_dir, self.temp_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_dir: str
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 100
    model_params: Dict[str, Any] = None
    model_list: List[str] = None
    feature_importance_threshold: float = 0.01
    prediction_threshold: float = 0.7

@dataclass
class AppConfig:
    """Main application configuration"""
    title: str = "AI Workforce Analysis"
    page_icon: str = "📊"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme: Dict[str, str] = None
    cache_ttl: int = 3600
    max_upload_size: int = 10 * 1024 * 1024  # 10MB

@dataclass
class DataConfig:
    """Data processing configuration"""
    schema: Any = HR_SCHEMA
    required_columns: Dict[str, str] = None
    date_columns: List[str] = None
    categorical_columns: List[str] = None
    numeric_columns: List[str] = None
    preprocessing_steps: List[str] = None
    validation_strict: bool = True
    max_missing_values: float = 0.1

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "app.log"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5

@dataclass
class Config:
    """Main configuration class"""
    app: AppConfig
    model: ModelConfig
    data: DataConfig
    paths: PathConfig
    logging: LoggingConfig
    api_key: Optional[str] = None
    debug: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        load_dotenv()
        
        # Create paths configuration
        paths = PathConfig()
        
        # Create model configuration
        model = ModelConfig(
            model_dir=paths.model_dir,
            model_list=["RandomForest", "XGBoost", "LightGBM"],
            model_params={
                "RandomForest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1
                },
                "XGBoost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1
                },
                "LightGBM": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1
                }
            }
        )
        
        # Create app configuration
        app = AppConfig(
            theme={
                'primaryColor': "#1f77b4",
                'backgroundColor': "#ffffff",
                'secondaryBackgroundColor': "#f0f2f6",
                'textColor': "#262730",
                'font': "sans serif"
            }
        )
        
        # Create data configuration
        data = DataConfig(
            required_columns=HR_SCHEMA.columns,
            date_columns=["HireDate", "TerminationDate"],
            categorical_columns=[
                "Department", "JobRole", "EducationField", "Gender",
                "MaritalStatus", "Attrition"
            ],
            numeric_columns=[
                "Age", "Salary", "YearsAtCompany", "JobSatisfaction",
                "WorkLifeBalance", "PerformanceRating", "Education",
                "NumCompaniesWorked", "TotalWorkingYears", "TrainingTimesLastYear",
                "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"
            ],
            preprocessing_steps=[
                "handle_missing_values",
                "convert_dates",
                "encode_categorical",
                "scale_numeric"
            ]
        )
        
        # Create logging configuration
        logging = LoggingConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            file=os.path.join(paths.log_dir, 'app.log')
        )
        
        return cls(
            app=app,
            model=model,
            data=data,
            paths=paths,
            logging=logging,
            api_key=os.getenv('OPENAI_API_KEY'),
            debug=os.getenv('DEBUG', 'False').lower() == 'true'
        )

    def save(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'app': self.app.__dict__,
            'model': self.model.__dict__,
            'data': {
                'required_columns': {k: v.dict() for k, v in self.data.required_columns.items()},
                'date_columns': self.data.date_columns,
                'categorical_columns': self.data.categorical_columns,
                'numeric_columns': self.data.numeric_columns,
                'preprocessing_steps': self.data.preprocessing_steps,
                'validation_strict': self.data.validation_strict,
                'max_missing_values': self.data.max_missing_values
            },
            'paths': self.paths.__dict__,
            'logging': self.logging.__dict__,
            'api_key': self.api_key,
            'debug': self.debug
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Create default configuration
config = Config.from_env() 