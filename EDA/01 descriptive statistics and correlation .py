from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from eda_common import load_dataset, output_dir


def main() -> None:
    df, cfg = load_dataset()
    out = output_dir()

    print("\n=== BASIC INFORMATION ===")
    print(f"Shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nData types:\n", df.dtypes)

    missing = df.isna().sum().to_frame("Missing_Count")
    missing["Missing_Pct"] = missing["Missing_Count"] / len(df) * 100
    missing = missing.sort_values("Missing_Pct", ascending=False)

    numeric = df.select_dtypes(include="number")
    descriptive = numeric.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T
    descriptive["median"] = numeric.median()
    descriptive["skewness"] = numeric.skew()
    descriptive["kurtosis"] = numeric.kurtosis()

    # Rainfall is zero-inflated - flag this explicitly rather than let it
    # sit silently in the skewness number. High skew here is EXPECTED,
    # not a data quality problem to fix.
    zero_rain_pct = (df["BMD_Rainfall"] == 0).mean() * 100
    print(f"\nPercentage of dry days (BMD_Rainfall = 0): {zero_rain_pct:.2f}%")
    print(f"BMD_Rainfall skewness: {descriptive.loc['BMD_Rainfall', 'skewness']:.2f} "
          f"(high positive skew expected - zero-inflated, heavy monsoon tail)")

    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")

    # Rainfall's relationship to its own lag features - a genuinely
    # useful number the generic template doesn't compute, since it
    # directly supports the "why lag features matter" argument.
    lag_corr = pearson.loc["BMD_Rainfall", ["Rain_lag1", "Rain_lag2", "Rain_7day_avg"]]
    print("\nCorrelation of BMD_Rainfall with its own lag features:\n", lag_corr)

    # Station-level frequency / summary (34 stations) kept in full here,
    # even though charts elsewhere will use the Region grouping for readability.
    station_summary = (
        df.groupby("Station")
        .agg(
            Records=("Station", "size"),
            Mean_Rainfall=("BMD_Rainfall", "mean"),
            Median_Rainfall=("BMD_Rainfall", "median"),
            Rainy_Day_Pct=("Rainy_Day", "mean"),
        )
        .sort_values("Mean_Rainfall", ascending=False)
    )
    station_summary["Rainy_Day_Pct"] *= 100

    result = out / "01_descriptive_statistics_and_correlation.xlsx"
    with pd.ExcelWriter(result, engine="openpyxl") as writer:
        descriptive.to_excel(writer, sheet_name="Numeric_Statistics")
        missing.to_excel(writer, sheet_name="Missing_Values")
        pearson.to_excel(writer, sheet_name="Pearson_Correlation")
        spearman.to_excel(writer, sheet_name="Spearman_Correlation")
        station_summary.to_excel(writer, sheet_name="Station_Summary")

    plt.figure(figsize=(12, 9))
    sns.heatmap(pearson, cmap="coolwarm", center=0, annot=False)
    plt.title("Rainfall Dataset — Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out / "01_correlation_heatmap.png", dpi=220)
    plt.close()

    print("\nTop 5 wettest stations:\n", station_summary.head())
    print("\nTop 5 driest stations:\n", station_summary.tail())
    print(f"\nSaved: {result}")


if __name__ == "__main__":
    main()