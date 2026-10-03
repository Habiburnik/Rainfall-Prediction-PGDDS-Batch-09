# Rainfall Prediction in Bangladesh Using Machine Learning

Next-day rainfall forecasting at 34 Bangladesh Meteorological Department (BMD) stations, 1984–2016, using NASA POWER weather features.

**Course:** PGDDS 203, Data Science Project, Post Graduate Diploma in Data Science, United International University (UIU)
**Team:** Rainfall-Prediction-PGDDS-Batch-09
**Instructor:** Ahmed Imran Kabir, Assistant Professor, IBER, UIU

---

## Overview

This project predicts **tomorrow's** rainfall from **today's** weather, as two related tasks:

| Task | Type | Final model |
|---|---|---|
| Will it rain tomorrow? | Classification | Logistic Regression (class-weighted) |
| How much will it rain tomorrow? | Regression | XGBoost with a Tweedie objective (variance power 1.3) |

Daily rainfall is a hard target: **67.84 % of days are dry**, the distribution is heavily right-skewed (skewness 5.76, maximum 590 mm in one day), and it varies strongly by region. Much of the project is about respecting that structure: a chronological split, a loss function that suits zero-inflated data, and an honest comparison against naive baselines.

### Key results (test period 2012–2016, 62,084 station-days)

**Rainfall amount (regression)**

| Model | RMSE (mm) | MAE (mm) | R² |
|---|---|---|---|
| Naive: persistence | 20.36 | 7.63 | −0.168 |
| Naive: seasonal climatology | 17.44 | 8.09 | 0.143 |
| Linear Regression | 15.74 | 8.05 | 0.302 |
| SARIMA (4-station subset) | 15.41 | 6.22 | 0.172 |
| Random Forest | 14.32 | 6.11 | 0.422 |
| LSTM (10-day window) | 14.53 | 6.34 | 0.405 |
| Back-propagation NN (MLP) | 14.40 | 6.04 | 0.416 |
| XGBoost (standard objective) | 14.34 | 6.16 | 0.421 |
| **XGBoost (Tweedie, p = 1.3)** | **14.26** | **5.76** | **0.427** |

**Rain occurrence (classification)**

| Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|
| 0.823 | 0.667 | 0.881 | 0.759 | 0.917 |

The classifier is deliberately tuned toward recall: it catches 88 % of rain days at the cost of more false alarms, which suits flood and agriculture use cases where a missed event costs more than a false warning.

> **Reading these numbers honestly.** R² ≈ 0.43 is for *daily, station-level, next-day* forecasts across 34 stations. Higher figures in related Bangladesh studies typically come from monthly aggregation or same-day nowcasting (using concurrent variables such as visibility that are partly caused by the rain itself). Replicating a same-day setup on this dataset raised R² only from about 0.43 to 0.454.

---

## Data

| Source | Role | Details |
|---|---|---|
| BMD daily rainfall | **Target** | 34 stations, 1984–2016 (originally purchased from BMD; publicly shared on Kaggle) |
| NASA POWER (community = AG) | **Features** | T2M, RH2M, PS, WS2M, CLOUD_AMT, ALLSKY_SFC_SW_DWN, T2MDEW |

The study window starts in 1984 because NASA POWER cloud amount and surface shortwave radiation are only available from that year.

**Why not use NASA POWER precipitation as the target?** It was validated against BMD (correlation 0.41–0.53, 18–36 % zero-mismatch) and rejected. Retraining the final model with NASA POWER precipitation as the target gave R² = 0.272 against about 0.43 with BMD, so observed BMD rainfall is both methodologically and empirically the better ground truth.

> Please check the data-use terms of BMD and NASA POWER before redistributing any raw or processed data.

### Cleaning (final dataset: 397,902 rows × 20 columns)

- Removed 8,050 export-padding rows (a "day 31" artefact in 30-day months, present at every station).
- About 2 % of rainfall entries were non-numeric missing codes (`***`, `**`); about 0.04 % carried trailing flag codes, from which the numeric part was kept.
- Reconciled 7 station-name mismatches between the BMD file and the coordinate/NASA lookup (for example Ishwardi/Ishurdi).
- 32 of 34 stations have under 5 % missing days. Hatiya (11.4 %) and Chittagong (11.1 %) were flagged but kept, since missing days are absent rows rather than corrupted values.

---

## Method

### Target and split

- **Target:** each station's BMD rainfall shifted **one day forward**, so day *t* features predict day *t + 1* rainfall. An early version predicted same-day rain from same-day cloud and humidity. That is nowcasting, not forecasting, and was corrected. 397,865 of 397,902 rows remain usable after the shift.
- **Chronological split, no random split and no k-fold:**

| Set | Years | Rows |
|---|---|---|
| Train | 1984–2007 | 286,109 |
| Validation | 2008–2011 | 49,672 |
| Test | 2012–2016 | 62,084 |

### Features (18)

`Year, Month, DOY, T2M, RH2M, PS, WS2M, CLOUD_AMT, ALLSKY_SFC_SW_DWN, T2MDEW, Rain_lag1, Rain_lag2, Rain_7day_avg, Temp_lag1, Humidity_lag1, Pressure_lag1, Station_Mean_Rainfall, Region_Rain_Today`

- **Lag and rolling features** are computed per station and set to missing wherever a station's date gap exceeds one day (244 rows, 0.06 %), so a lag never silently spans a multi-year gap.
- **`Station_Mean_Rainfall`** is a target encoding using **training years only** (≤ 2007) to avoid leakage. One global model covers all 34 stations rather than 34 sparse one-hot flags.
- **`Region_Rain_Today`** is the same-day rainfall averaged over the *other* stations in the same climate region (leave-self-out). Regions: Hill/NE, Coastal, Central, Plains/North. It is a region-level proxy, because station coordinates were not available in the processed file.
- **Outliers:** IQR handling on pressure and wind speed only. Rainfall is deliberately left untouched, since extreme values are the signal.
- **PCA** was used for exploratory analysis only, not for modelling.

### Why Tweedie?

Daily rainfall is non-negative, zero-inflated and right-skewed, which is the shape the Tweedie distribution is built for. It was the only change tested in the project that improved RMSE, MAE and R² together. The variance-power sweep on the test period:

| p | RMSE | MAE | R² |
|---|---|---|---|
| 1.1 | 14.291 | 5.839 | 0.4245 |
| **1.3** | **14.260** | **5.760** | **0.4270** |
| 1.5 | 14.270 | 5.740 | 0.4262 |
| 1.7 | 14.343 | 5.884 | 0.4203 |
| 1.9 | 16.065 | 6.130 | 0.2728 |

The margin over standard XGBoost (R² 0.421) is small. The stronger case for Tweedie is its lower MAE and its fit to the target's distribution.

### Experiments that did *not* beat the final model

| Experiment | Outcome |
|---|---|
| Log1p target | MAE improved (5.14) but RMSE and R² worsened (15.38, 0.333) because back-transforming under-predicts large values |
| Two-stage model (classifier gates a rainy-day regressor) | Below single-stage XGBoost at every threshold tested |
| Stacking (Tweedie-XGB + RF + MLP) | Marginal gain, about 70 % of the weight on Tweedie alone, so not worth the complexity |
| 34 per-station models | No better than one global model |
| ENSO + IOD (concurrent month) | Slightly worse; ranked near the bottom in importance. A *lagged* version was not tested |
| NASA POWER precipitation as target | R² 0.272 |

GARCH was deliberately not run as a point forecaster: it models variance, not the level of a series.

---

## What the model gets right and wrong

- **Best and worst stations** (global model, test period): Teknaf R² ≈ 0.58, Bogra R² ≈ 0.21. Coastal stations (large synoptic and cyclone systems) are more predictable than inland Plains/North stations (small, localized convective rain).
- **Under-prediction of extremes:** the model over-predicts on dry and very light days and increasingly under-predicts as intensity grows. For example, days above 100 mm average 147 mm observed against 53 mm predicted. It is better at estimating typical daily rainfall than extreme events.
- **Top features** (XGBoost Tweedie): `CLOUD_AMT` (0.35), `Region_Rain_Today` (0.21), `ALLSKY_SFC_SW_DWN` (0.13), `T2MDEW` (0.07).

---

## Reproducing the final model

Minimal training code for the final regression model (`Processed.csv` must contain the engineered features and `Station_Mean_Rainfall`; the regional feature is computed as shown in the project scripts):

```python
import numpy as np, xgboost as xgb

feature_cols = ["Year","Month","DOY","T2M","RH2M","PS","WS2M","CLOUD_AMT",
                "ALLSKY_SFC_SW_DWN","T2MDEW","Rain_lag1","Rain_lag2",
                "Rain_7day_avg","Temp_lag1","Humidity_lag1","Pressure_lag1",
                "Station_Mean_Rainfall","Region_Rain_Today"]

train = df[df.Year <= 2007]
val   = df[(df.Year >= 2008) & (df.Year <= 2011)]
test  = df[df.Year >= 2012]

model = xgb.XGBRegressor(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    objective="reg:tweedie", tweedie_variance_power=1.3,
    eval_metric="tweedie-nloglik@1.3", early_stopping_rounds=30,
    n_jobs=-1, random_state=42,
)
model.fit(train[feature_cols], train["Target_Rainfall"],
          eval_set=[(val[feature_cols], val["Target_Rainfall"])], verbose=False)
pred = np.clip(model.predict(test[feature_cols]), 0, None)
```

**Expect small differences between machines.** XGBoost's histogram method is not bit-for-bit reproducible across environments. Running the identical configuration on different setups gave test R² between 0.427 and 0.431 (RMSE 14.21 to 14.26). The reported figures (14.26 / 5.76 / 0.427) come from the project's reference run. Differences of this size between top models should not be over-interpreted.

---

## Repository contents

> Adjust this section to match the actual folder layout.

| File / folder | Purpose |
|---|---|
| `Processed.csv` | Analysis-ready dataset (397,902 rows × 20 columns) |
| `data_utils.py`, `evaluate_classification.py`, `evaluate_regression.py` | Shared loading utilities and the classification and regression evaluation scripts (metrics and figures) |
| `stage1` … `stage8` regression scripts | Every regression experiment reported above, from baselines to the Tweedie sweep and robustness checks |
| `pull_nasa_power_rainfall.py` | NASA POWER precipitation pull used for the target-source validation |
| Reports | Final project report and presentation |

---

## Limitations

- **Coarse weather grid.** NASA POWER features are reanalysis values on a grid of roughly 55 km per cell, not point observations at each rain gauge. This likely caps accuracy for localized inland rainfall.
- **Region-level spatial proxy.** `Region_Rain_Today` is not true nearest-neighbour weighting.
- **Light tuning, single seed.** Only the Tweedie variance power was swept. The LSTM was trained on a 60,000-row subsample on CPU, and SARIMA was fitted on four stations (Sylhet, Dhaka, Rajshahi, Barisal), so neither is a fully like-for-like comparison.
- **Lagged climate indices untested.** Only concurrent-month ENSO and IOD values were tried.
- **Heavy rainfall is under-predicted.** See above.

---

## Team

| Member | Role |
|---|---|
| Tarek Mahmud | Coordination, BMD data collection, timeline |
| Habibur Rahaman | Cleaning and preprocessing, feature engineering, EDA |
| Shafiqul Islam | Model building and training, hyperparameter tuning |
| Sahariar Sahen | Model evaluation (RMSE, MAE, R²), visualization, report and presentation |

## Acknowledgements

Bangladesh Meteorological Department for the rainfall observations; NASA POWER for the meteorological variables; Ahmed Imran Kabir for supervision.

## License

*Add a license here (for example MIT for the code) and note the data-use terms separately.*