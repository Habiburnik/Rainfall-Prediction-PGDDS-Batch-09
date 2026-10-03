from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

from eda_common import load_dataset, output_dir


def iqr_flag(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (series < q1 - multiplier * iqr) | (series > q3 + multiplier * iqr)


def zscore_flag(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    return ((series - series.mean()) / std).abs() > threshold


def main() -> None:
    df, cfg = load_dataset()
    out = output_dir()

    # Weather features only - BMD_Rainfall is deliberately excluded, as
    # established in the Preprocessing Report: extreme rainfall is the
    # signal this project models, not noise to flag for removal.
    anomaly_cols = cfg["anomaly_cols"]

    flags = df.copy()
    iqr_cols, z_cols = [], []
    for col in anomaly_cols:
        iqr_name = f"IQR_Anomaly_{col}"
        z_name = f"Z_Anomaly_{col}"
        flags[iqr_name] = iqr_flag(flags[col])
        flags[z_name] = zscore_flag(flags[col])
        iqr_cols.append(iqr_name)
        z_cols.append(z_name)
    flags["Any_IQR_Anomaly"] = flags[iqr_cols].any(axis=1)
    flags["Any_Z_Anomaly"] = flags[z_cols].any(axis=1)

    # --- Multivariate anomaly detection: Isolation Forest ---
    # This is genuinely new analysis, distinct from the earlier univariate
    # IQR check - it can flag a row where NO single feature is individually
    # extreme, but the COMBINATION of values is unusual (e.g. high temp +
    # high pressure + high humidity together, an unlikely joint occurrence).
    model_data = df[anomaly_cols].copy()
    model_data = model_data.fillna(model_data.median(numeric_only=True))
    X = StandardScaler().fit_transform(model_data)

    model = IsolationForest(n_estimators=250, contamination=0.02, random_state=42, n_jobs=-1)
    predictions = model.fit_predict(X)
    flags["Isolation_Forest_Anomaly"] = predictions == -1
    flags["Isolation_Forest_Score"] = model.decision_function(X)

    summary = pd.DataFrame({
        "Method": ["Any IQR rule (per-feature)", "Any Z-score rule (per-feature)", "Isolation Forest (multivariate)"],
        "Anomaly_Count": [
            int(flags["Any_IQR_Anomaly"].sum()),
            int(flags["Any_Z_Anomaly"].sum()),
            int(flags["Isolation_Forest_Anomaly"].sum()),
        ],
    })
    summary["Anomaly_Pct"] = summary["Anomaly_Count"] / len(df) * 100

    # Rows flagged by BOTH univariate rules AND Isolation Forest - highest
    # priority for manual review, per the course's own guidance.
    high_priority = flags[
        (flags["Any_IQR_Anomaly"] | flags["Any_Z_Anomaly"]) & flags["Isolation_Forest_Anomaly"]
    ]

    result = out / "05_anomaly_analysis.xlsx"
    with pd.ExcelWriter(result, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Anomaly_Summary", index=False)
        flags.loc[flags["Isolation_Forest_Anomaly"]].to_excel(writer, sheet_name="IsolationForest_Flagged", index=False)
        high_priority.to_excel(writer, sheet_name="High_Priority_Both_Methods", index=False)

    # Visualize: two features where Isolation Forest disagrees most from
    # a simple univariate view - use RH2M vs T2M, a physically meaningful pair
    plot_data = flags[["RH2M", "T2M", "Isolation_Forest_Anomaly"]].dropna()
    plot_data = plot_data.sample(min(4000, len(plot_data)), random_state=42)
    normal = plot_data[~plot_data["Isolation_Forest_Anomaly"]]
    anomalous = plot_data[plot_data["Isolation_Forest_Anomaly"]]
    plt.figure(figsize=(8, 6))
    plt.scatter(normal["RH2M"], normal["T2M"], alpha=0.25, s=10, label="Normal")
    plt.scatter(anomalous["RH2M"], anomalous["T2M"], marker="x", s=40, color="red", label="Isolation Forest Anomaly")
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("Temperature (°C)")
    plt.title("Isolation Forest Multivariate Anomalies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "05_isolation_forest_anomalies.png", dpi=600)
    plt.close()

    print("=== ANOMALY SUMMARY ===")
    print(summary)
    print(f"\nRows flagged by BOTH a univariate rule AND Isolation Forest: {len(high_priority)}")
    print(f"\nSaved: {result}")


if __name__ == "__main__":
    main()