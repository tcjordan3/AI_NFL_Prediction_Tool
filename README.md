# AI_NFL_Prediction_Tool

Explore historical and predicted NFL player data. Predictions generated via XGBoost

TODO: insert sample

## Acknowledgements

- **[Pro Football Reference](https://www.pro-football-reference.com/)** — Historical player stats

- **[nflfastR](https://www.nflfastr.com/)** — Play-by-play data and advanced metrics

## Model Details

XGBoost model trained for the following targets: `[g, yds, att, cmp, td, int]`. Model accepts per-target feature input and per-target hyperparameters. Validated via cross-fold validation and SHAP analysis. Enhanced via hyperparameter tuning and recursive feature elimination

## Known Limitations

- Limited data size prevents extensive cross-fold validation

- Certain prediction combinations result in infeasible data (i.e cmp% > 100)

- Model lacks knowledge of up-to-date rosters, including retirements and projected starters

- Model cannot generate predictions for rookies

## Future Extensions

- Introduce team-level features (i.e Off/Dec Rank, PF/PA, etc.)

- Collect data and generate predictions for RB/WR/TE

- Forecast multiple years into the future

## Installation

### Prerequisites
- Python >= 3.11

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/tcjordan3/AI_NFL_Prediction_Tool.git
cd AI_NFL_Prediction_Tool
```

2. **Create a virtual environment**

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\\Scripts\\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install the package**
```bash
pip install .
```

### Running the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`
