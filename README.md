# AI Workforce Analysis App

A Streamlit application for analyzing workforce data and predicting employee attrition using AI.

## Features

- Data validation and preprocessing
- Interactive data visualization
- Attrition risk prediction
- Feature importance analysis
- High-risk employee identification
- Export capabilities

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

3. Upload your HR data in CSV format

4. Explore the analysis and visualizations

## Project Structure

```
workforce_analysis_app/
├── app.py                 # Main Streamlit application
├── config/               # Configuration files
│   ├── config.py        # Configuration management
│   └── default_config.yaml
├── agents/              # AI analysis agents
│   ├── base_agent.py
│   └── attrition_agent.py
├── schemas/             # Data validation schemas
│   └── data_schema.py
├── utils/              # Utility functions
│   ├── data_loader.py
│   └── logger.py
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Configuration

The application uses a YAML-based configuration system. You can modify settings in `config/default_config.yaml` or override them using environment variables.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 