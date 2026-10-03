# Rainfall Prediction for Bangladesh using Machine Learning
### PGDDS Batch 09 – United International University (UIU)

An end-to-end machine learning project for predicting daily rainfall across Bangladesh using historical meteorological observations.

The project combines NASA POWER weather data with Bangladesh Meteorological Department (BMD) rainfall observations to build a clean, modeling-ready dataset, perform exploratory data analysis, engineer predictive features, and develop rainfall prediction models.

---

## Project Overview

Rainfall prediction is one of the most challenging regression problems because rainfall is highly seasonal, zero-inflated, and influenced by complex atmospheric processes.

Instead of relying solely on satellite-estimated rainfall, this project validates multiple data sources and uses:

- **NASA POWER** for meteorological features
- **Bangladesh Meteorological Department (BMD)** as the ground-truth rainfall source

The objective is to build a reliable machine learning pipeline capable of predicting **daily rainfall** for a selected station and date.

---

## Objectives

- Collect historical weather data from NASA POWER
- Integrate weather observations with BMD rainfall records
- Perform data cleaning and preprocessing
- Engineer meaningful temporal and station-based features
- Analyze rainfall characteristics across Bangladesh
- Train and evaluate machine learning regression models
- Compare model performance against a simple baseline

---

## Data Sources

### NASA POWER

Daily weather variables:

- Temperature (T2M)
- Maximum Temperature (T2M_MAX)
- Minimum Temperature (T2M_MIN)
- Relative Humidity (RH2M)
- Surface Pressure (PS)
- Wind Speed (WS2M)
- Dew Point Temperature (T2MDEW)
- Cloud Amount (CLOUD_AMT)
- Solar Radiation (ALLSKY_SFC_SW_DWN)

### Bangladesh Meteorological Department (BMD)

- Daily station rainfall observations
- Used as the prediction target (ground truth)

---

## Machine Learning Features

### Weather Features

- Temperature
- Humidity
- Pressure
- Wind Speed
- Cloud Amount
- Solar Radiation
- Dew Point Temperature

### Temporal Features

- Year
- Month
- Day of Year (DOY)

### Engineered Features

- Rain_lag1
- Rain_lag2
- Rain_7day_avg
- Temp_lag1
- Humidity_lag1
- Pressure_lag1
- Station_Mean_Rainfall

---

## Project Workflow

```text
NASA POWER Data
        │
        ▼
Download Weather Data
        │
        ▼
Merge with BMD Rainfall
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Trend Analysis
        │
        ▼
Anomaly Detection
        │
        ▼
Model Training
        │
        ▼
Model Evaluation
```

---

## Repository Structure

```
Rainfall-Prediction-PGDDS-Batch-09
│
├── Data
│   ├── Raw
│   ├── Merged
│   ├── Processed
│   └── EDA_output
│
├── EDA
│   ├── eda_common.py
│   ├── 01_descriptive_statistics.py
│   ├── 02_univariate_analysis.py
│   ├── 03_bivariate_analysis.py
│   ├── 04_trend_analysis.py
│   └── 05_anomaly_detection.py
│
├── CollectDataFromNasaPower.ipynb
├── Data_Merge.ipynb
├── Data_Preprocessing.ipynb
├── Data_Cleaning_Scoring.ipynb
├── requirements.txt
└── README.md
```

---

## Exploratory Data Analysis

The EDA pipeline includes:

- Descriptive statistics
- Missing value analysis
- Pearson correlation
- Spearman correlation
- Distribution analysis
- Rainfall seasonality
- Regional comparison
- Trend analysis
- Anomaly detection
- Station-level summaries

Outputs are exported as:

- Excel reports
- CSV summaries
- Publication-quality figures

---

## Machine Learning Pipeline

The preprocessing workflow uses:

- Missing value imputation
- One-Hot Encoding
- Feature Scaling
- PCA (for dimensionality analysis)
- Feature Engineering
- Train/Test Split

Models evaluated include:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost
- LightGBM
- CatBoost

---

## Evaluation Metrics

Model performance is evaluated using:

- R² Score
- RMSE
- MAE

Performance is compared against a simple historical baseline to ensure the machine learning model provides meaningful improvement.

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Rainfall-Prediction-PGDDS-Batch-09
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the workflow in the following order:

1. Collect NASA POWER weather data
2. Merge weather and rainfall datasets
3. Preprocess and engineer features
4. Run exploratory data analysis
5. Train machine learning models
6. Evaluate prediction performance

---

## Output

The project generates:

- Clean modeling-ready dataset
- Descriptive statistics
- Correlation matrices
- Seasonal rainfall analysis
- Trend reports
- Anomaly reports
- Visualization figures
- Model evaluation results

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- OpenPyXL
- XGBoost
- LightGBM
- CatBoost
- Jupyter Notebook

---

## Academic Information

**Program:** Post Graduate Diploma in Data Science (PGDDS)

**Institution:** United International University (UIU)

**Project Type:** Academic Machine Learning Project

---

## License

This project is developed for academic and educational purposes.