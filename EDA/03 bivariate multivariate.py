from __future__ import annotations

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

from eda_common import load_dataset, output_dir


def main() -> None:
    df, cfg = load_dataset()
    out = output_dir()
    sns.set_theme(style="whitegrid")

    # --- 1. Humidity vs Rainfall scatter, colored by Rainy_Day ---
    sample = df[["RH2M", "BMD_Rainfall", "Rainy_Day"]].dropna()
    sample = sample.sample(min(5000, len(sample)), random_state=42)
    plt.figure(figsize=(8, 6))
    for label, color in [(False, "#0A7B83"), (True, "#C0392B")]:
        part = sample[sample["Rainy_Day"] == label]
        plt.scatter(part["RH2M"], part["BMD_Rainfall"], alpha=0.3, s=12,
                    color=color, label="Rain" if label else "No Rain")
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("Rainfall (mm)")
    plt.title("Humidity vs Rainfall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "03_scatter_humidity_vs_rainfall.png", dpi=600)
    plt.close()

    # --- 2. Rainy-Day rate by Region ---
    region_rate = (
    df.groupby("Region")["Rainy_Day"].mean().sort_values(ascending=False) * 100
    )

    plt.figure(figsize=(8, 5))

    region_rate.plot(
    kind="bar",
    color="steelblue",
    edgecolor="black",
    width=0.35
    )

    plt.ylabel("Rainy-Day Rate (%)")
    plt.title("Percentage of Days with Measurable Rain, by Region")
    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(out / "03_rainy_day_rate_by_region.png", dpi=600)
    plt.close()

    # --- 3. Monthly seasonality: rainfall vs temperature (dual axis) ---
    monthly = df.groupby("Month").agg(
        Mean_Rainfall=("BMD_Rainfall", "mean"),
        Mean_Temp=("T2M", "mean"),
    )
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.bar(monthly.index, monthly["Mean_Rainfall"], color="#0A7B83", alpha=0.7, label="Mean Rainfall")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Mean Rainfall (mm)", color="#0A7B83")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index, monthly["Mean_Temp"], color="#C0392B", marker="o", linewidth=2, label="Mean Temp")
    ax2.set_ylabel("Mean Temperature (°C)", color="#C0392B")
    plt.title("Monthly Seasonality: Rainfall vs Temperature")
    ax1.set_xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(out / "03_monthly_seasonality_rain_temp.png", dpi=600)
    plt.close()

    # --- 4. Rainy-Day rate heatmap: Region x Month ---
    pivot = (
        df.groupby(["Region", "Month"])["Rainy_Day"].mean().unstack() * 100
    )
    plt.figure(figsize=(11, 5))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".0f", cbar_kws={"label": "Rainy-Day Rate (%)"})
    plt.title("Rainy-Day Rate (%) by Region and Month")
    plt.xlabel("Month")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(out / "03_heatmap_region_month_rainy_rate.png", dpi=600)
    plt.close()

    print("Region rainy-day rates:\n", region_rate)
    print("\nBivariate/multivariate charts saved in:", out)


if __name__ == "__main__":
    main()