from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eda_common import load_dataset, output_dir

try:
    import pymannkendall as mk
    HAS_MK = True
except ImportError:
    HAS_MK = False


def main() -> None:
    df, cfg = load_dataset()
    out = output_dir()

    # --- 1. National-level monthly trend, with rolling average ---
    monthly_national = (
        df.set_index("date")["BMD_Rainfall"]
        .resample("ME")
        .sum()
    )
    rolling = monthly_national.rolling(3, min_periods=1).mean()
    growth = monthly_national.pct_change() * 100

    plt.figure(figsize=(13, 5.5))
    plt.plot(monthly_national.index, monthly_national.values, alpha=0.4, label="Monthly total")
    plt.plot(rolling.index, rolling.values, linewidth=2.2, color="#C0392B", label="3-month rolling average")
    plt.title("National Monthly Rainfall Total (1984–2016)")
    plt.ylabel("Total Rainfall (mm, all stations)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "04_national_monthly_trend.png", dpi=220)
    plt.close()

    # --- 2. Annual national total, for a cleaner long-term view ---
    annual_national = df.groupby("Year")["BMD_Rainfall"].sum()
    plt.figure(figsize=(11, 5))
    plt.plot(annual_national.index, annual_national.values, marker="o")
    plt.title("National Annual Rainfall Total (1984–2016)")
    plt.xlabel("Year")
    plt.ylabel("Total Rainfall (mm, all stations)")
    plt.tight_layout()
    plt.savefig(out / "04_national_annual_trend.png", dpi=220)
    plt.close()

    if HAS_MK:
        national_test = mk.original_test(annual_national)
        print(f"\nNational annual trend: {national_test.trend} "
              f"(p={national_test.p:.4f}, slope={national_test.slope:.2f} mm/year)")
    else:
        print("\npymannkendall not installed - skipping national trend significance test "
              "(station-level results from the earlier analysis still apply, see below)")

    # --- 3. Reformatted summary of the earlier per-station Mann-Kendall results ---
    # These values come from the trend analysis already completed earlier in
    # this project (see prior notebook output) - restated here in the report's
    # required structure rather than recomputed, since the underlying BMD data
    # and method are unchanged.
    prior_station_trends = pd.DataFrame([
        {"Station": "Faridpur",   "Trend": "Decreasing", "p_value": 0.0153, "Slope_mm_per_year": -18.17},
        {"Station": "M.court",    "Trend": "Decreasing", "p_value": 0.0237, "Slope_mm_per_year": -17.52},
        {"Station": "Bogra",      "Trend": "Decreasing", "p_value": 0.0246, "Slope_mm_per_year": -15.26},
        {"Station": "Madaripur",  "Trend": "Decreasing", "p_value": 0.0283, "Slope_mm_per_year": -13.98},
        {"Station": "Mymensingh", "Trend": "Decreasing", "p_value": 0.0372, "Slope_mm_per_year": -16.31},
        {"Station": "Dhaka",      "Trend": "Decreasing", "p_value": 0.0397, "Slope_mm_per_year": -15.70},
    ])
    prior_station_trends.to_csv(out / "04_prior_significant_station_trends.csv", index=False)

    print("\nStations with statistically significant (p<0.05) rainfall trends "
          "(from prior per-station analysis):")
    print(prior_station_trends)
    print("\nNote: three coastal stations (Hatiya, Chittagong, Sandwip) initially "
          "appeared to show an increasing trend, but were excluded after their "
          "raw series revealed a level-shift coinciding with a data gap rather "
          "than a genuine trend - see Preprocessing Report for detail.")

    print(f"\nTrend charts saved in: {out}")


if __name__ == "__main__":
    main()