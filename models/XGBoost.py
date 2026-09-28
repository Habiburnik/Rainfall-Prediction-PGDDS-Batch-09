from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = Path("./Data/Processed/Processed.csv")

OUTPUT_DIR = Path("outputs/xgboost_regression")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("XGBOOST NEXT-DAY RAINFALL REGRESSION")
print("=" * 70)

print("\nLoading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# 3. DATE PROCESSING
# ============================================================

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

df = df.dropna(
    subset=["date"]
)


# ============================================================
# 4. SORT BY STATION AND DATE
# ============================================================

df = (
    df.sort_values(
        ["Station", "date"]
    )
    .reset_index(drop=True)
)


# ============================================================
# 5. CREATE VALID NEXT-DAY RAINFALL TARGET
# ============================================================

print("\nCreating next-day rainfall target...")

# Next available date for each station
df["next_date"] = (
    df.groupby("Station")["date"].shift(-1)
)

# Number of days between current and next record
df["days_to_next"] = (
    df["next_date"] - df["date"]
).dt.days

# Next day's actual rainfall
df["Target_Rainfall"] = (
    df.groupby("Station")["BMD_Rainfall"].shift(-1)
)

# Only accept the target if the next record
# is exactly one calendar day later
df.loc[
    df["days_to_next"] != 1,
    "Target_Rainfall"
] = np.nan


print(
    "Rows with valid next-day target:",
    df["Target_Rainfall"].notna().sum()
)


# ============================================================
# 6. REMOVE INVALID TARGET ROWS
# ============================================================

df = df.dropna(
    subset=["Target_Rainfall"]
).copy()


# ============================================================
# 7. FEATURE SELECTION
# ============================================================

numeric_features = [
    "Year",
    "Month",
    "DOY",

    "T2M",
    "RH2M",
    "PS",
    "WS2M",
    "CLOUD_AMT",
    "ALLSKY_SFC_SW_DWN",
    "T2MDEW",

    "BMD_Rainfall",

    "Rain_lag1",
    "Rain_lag2",
    "Rain_7day_avg",

    "Temp_lag1",
    "Humidity_lag1",
    "Pressure_lag1",
]

categorical_features = [
    "Station"
]


required_features = (
    numeric_features
    + categorical_features
)


# ============================================================
# 8. CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    col
    for col in required_features
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ============================================================
# 9. PREPARE MODEL DATA
# ============================================================

model_df = df[
    required_features
    + [
        "Target_Rainfall",
        "date"
    ]
].copy()


print("\nMissing values:")

print(
    model_df.isna()
    .sum()
    .sort_values(
        ascending=False
    )
)


# Remove rows with missing feature values
model_df = model_df.dropna(
    subset=required_features
).copy()


# ============================================================
# 10. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 70)

# Training: 1984 - 2010
# Testing : 2011 - 2016

train_df = model_df[
    model_df["date"] < "2011-01-01"
].copy()

test_df = model_df[
    model_df["date"] >= "2011-01-01"
].copy()


print(
    f"\nTraining rows: {len(train_df):,}"
)

print(
    f"Testing rows : {len(test_df):,}"
)

print(
    f"\nTraining period:"
    f" {train_df['date'].min().date()}"
    f" → {train_df['date'].max().date()}"
)

print(
    f"Testing period:"
    f" {test_df['date'].min().date()}"
    f" → {test_df['date'].max().date()}"
)


# ============================================================
# 11. X AND Y
# ============================================================

X_train = train_df[
    required_features
]

y_train = train_df[
    "Target_Rainfall"
]

X_test = test_df[
    required_features
]

y_test = test_df[
    "Target_Rainfall"
]


# ============================================================
# 12. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",

            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),

            categorical_features
        ),

        (
            "numeric",

            "passthrough",

            numeric_features
        )
    ]
)


# ============================================================
# 13. XGBOOST REGRESSOR
# ============================================================

model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    eval_metric="rmse",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 14. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )
    ]
)


# ============================================================
# 15. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST REGRESSOR")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)

print("\nTraining completed.")


# ============================================================
# 16. ACTUAL PREDICTION
# ============================================================

print("\nGenerating next-day rainfall predictions...")

y_pred = pipeline.predict(
    X_test
)


# ============================================================
# 17. MAKE SURE PREDICTIONS ARE NOT NEGATIVE
# ============================================================

# Rainfall cannot physically be negative.

y_pred = np.maximum(
    y_pred,
    0
)


# ============================================================
# 18. EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("XGBOOST REGRESSION RESULTS")
print("=" * 70)

print(
    f"\nMAE  : {mae:.4f} mm"
)

print(
    f"RMSE : {rmse:.4f} mm"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# 19. ACTUAL VS PREDICTED TABLE
# ============================================================

results = test_df[
    [
        "date",
        "Station",
        "BMD_Rainfall"
    ]
].copy()

results["Actual_Next_Day_Rainfall"] = (
    y_test.values
)

results["Predicted_Next_Day_Rainfall"] = (
    y_pred
)

results["Absolute_Error"] = (
    np.abs(
        results["Actual_Next_Day_Rainfall"]
        -
        results["Predicted_Next_Day_Rainfall"]
    )
)


print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED")
print("=" * 70)

print(
    results[
        [
            "date",
            "Station",
            "Actual_Next_Day_Rainfall",
            "Predicted_Next_Day_Rainfall",
            "Absolute_Error"
        ]
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE PREDICTIONS
# ============================================================

results.to_csv(

    OUTPUT_DIR
    / "actual_vs_predicted.csv",

    index=False
)


# ============================================================
# 21. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],

    "Value": [
        mae,
        rmse,
        r2
    ]
})

metrics.to_csv(

    OUTPUT_DIR
    / "regression_metrics.csv",

    index=False
)


# ============================================================
# 22. ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_test,
    y_pred,
    alpha=0.25,
    s=10
)

# Perfect prediction line
max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel(
    "Actual Next-Day Rainfall (mm)"
)

plt.ylabel(
    "Predicted Next-Day Rainfall (mm)"
)

plt.title(
    "XGBoost: Actual vs Predicted Next-Day Rainfall"
)

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR
    / "actual_vs_predicted_scatter.png",

    dpi=250
)

plt.close()


# ============================================================
# 23. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

fitted_preprocessor = (
    pipeline
    .named_steps["preprocessor"]
)

xgb_model = (
    pipeline
    .named_steps["model"]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance = (
    xgb_model.feature_importances_
)

feature_importance = pd.DataFrame({

    "Feature": feature_names,

    "Importance": importance
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    feature_importance
    .head(20)
    .to_string(
        index=False
    )
)


# Save
feature_importance.to_csv(

    OUTPUT_DIR
    / "feature_importance.csv",

    index=False
)


# ============================================================
# 24. FEATURE IMPORTANCE PLOT
# ============================================================

top_features = (
    feature_importance
    .head(20)
    .sort_values(
        "Importance"
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Regression - Top 20 Features"
)

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR
    / "feature_importance.png",

    dpi=250
)

plt.close()


# ============================================================
# 25. SAVE MODEL
# ============================================================

joblib.dump(

    pipeline,

    OUTPUT_DIR
    / "xgboost_rainfall_regression.pkl"
)


# ============================================================
# 26. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"\nMAE  : {mae:.4f} mm"
)

print(
    f"RMSE : {rmse:.4f} mm"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    f"\nOutputs saved to:"
    f"\n{OUTPUT_DIR.resolve()}"
)

print("\nFiles generated:")

print(" - actual_vs_predicted.csv")
print(" - regression_metrics.csv")
print(" - actual_vs_predicted_scatter.png")
print(" - feature_importance.csv")
print(" - feature_importance.png")
print(" - xgboost_rainfall_regression.pkl")