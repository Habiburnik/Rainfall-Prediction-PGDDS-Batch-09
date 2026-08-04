from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eda_common import load_dataset, output_dir


def main() -> None:
    df, cfg = load_dataset()
    out = output_dir()

    # --- 1. Rainfall distribution: raw (shows zero-inflation) ---
    plt.figure(figsize=(9, 5.5))
    plt.hist(df["BMD_Rainfall"], bins=60, edgecolor="black", alpha=0.8)
    plt.title("Distribution of Daily Rainfall (raw) — 67.8% of days are dry")
    plt.xlabel("Rainfall (mm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out / "02_hist_rainfall_raw.png", dpi=220)
    plt.close()

    # --- 2. Rainfall distribution: rainy days only, log scale ---
    # The raw histogram is dominated by the zero-bar; this second view
    # shows the shape of rainfall AMOUNT on days it actually rains,
    # which the raw histogram hides entirely.
    rainy = df.loc[df["Rainy_Day"], "BMD_Rainfall"]
    plt.figure(figsize=(9, 5.5))
    plt.hist(np.log1p(rainy), bins=50, edgecolor="black", alpha=0.8, color="steelblue")
    plt.title("Distribution of Rainfall on Rainy Days Only (log(1+mm) scale)")
    plt.xlabel("log(1 + Rainfall mm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out / "02_hist_rainfall_rainy_days_log.png", dpi=220)
    plt.close()

    # --- 3. Boxplot of rainfall by Region (readable grouping) ---
    clean = df[["Region", "BMD_Rainfall"]].dropna()
    grouped = list(clean.groupby("Region"))
    values = [part["BMD_Rainfall"].values for _, part in grouped]
    labels = [str(name) for name, _ in grouped]
    plt.figure(figsize=(9, 6))
    plt.boxplot(values, tick_labels=labels, showfliers=False)  # outliers hide the box at this scale
    plt.title("Rainfall by Region (outliers excluded from view only, not from data)")
    plt.ylabel("Rainfall (mm)")
    plt.tight_layout()
    plt.savefig(out / "02_boxplot_rainfall_by_region.png", dpi=220)
    plt.close()

    # --- 4. Weather feature distributions (2x4 grid) ---
    weather_cols = ["T2M", "RH2M", "PS", "WS2M", "CLOUD_AMT", "ALLSKY_SFC_SW_DWN", "T2MDEW"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, col in zip(axes.flat, weather_cols):
        ax.hist(df[col].dropna(), bins=40, edgecolor="black", alpha=0.75, color="steelblue")
        ax.set_title(col, fontsize=10)
    axes.flat[-1].axis("off")  # 7 features in an 8-slot grid
    plt.suptitle("Distribution of Weather Features")
    plt.tight_layout()
    plt.savefig(out / "02_weather_feature_distributions.png", dpi=200)
    plt.close()

    print(f"Univariate charts saved in: {out}")
    print(f"Rainfall (rainy days only) — mean: {rainy.mean():.2f} mm, "
          f"median: {rainy.median():.2f} mm, max: {rainy.max():.1f} mm")


if __name__ == "__main__":
    main()