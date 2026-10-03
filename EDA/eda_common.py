from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_PATH = Path("Data/Processed/Processed.csv")
OUTPUT_ROOT = Path("outputs/Eda")

# Region grouping - used ONLY for chart readability (34 stations is too
# many for a clean x-axis). Full station-level detail is preserved in
# all tables/CSVs; only visuals aggregate to region.
REGION_MAP = {
    "Sylhet": "Hill/Northeast", "Srimangal": "Hill/Northeast", "Rangamati": "Hill/Northeast",
    "Chittagong": "Coastal", "CoxsBazar": "Coastal", "Teknaf": "Coastal", "Sitakunda": "Coastal",
    "Sandwip": "Coastal", "Kutubdia": "Coastal", "Hatiya": "Coastal", "Khepupara": "Coastal",
    "Patuakhali": "Coastal", "Bhola": "Coastal", "Barisal": "Coastal", "M.court": "Coastal",
    "Mongla": "Coastal", "Khulna": "Coastal", "Satkhira": "Coastal",
    "Dhaka": "Central", "Faridpur": "Central", "Madaripur": "Central", "Tangail": "Central",
    "Comilla": "Central", "Chandpur": "Central", "Feni": "Central",
    "Rajshahi": "Plains/North", "Bogra": "Plains/North", "Rangpur": "Plains/North",
    "Dinajpur": "Plains/North", "sydpur": "Plains/North", "Ishurdi": "Plains/North",
    "Jessore": "Plains/North", "chuadanga": "Plains/North", "Mymensingh": "Plains/North",
}

CFG = {
    "date_col": "date",
    "group_col": "Station",
    "region_col": "Region",
    "primary_metric": "BMD_Rainfall",
    "secondary_metric": "T2M",
    "target_col": "Rainy_Day",       # derived, see load_dataset()
    "positive_label": True,
    "x_numeric": "RH2M",
    "y_numeric": "BMD_Rainfall",
    "trend_metric": "BMD_Rainfall",
    "trend_agg": "sum",
    "trend_freq": "ME",              # monthly - matches monsoon seasonality framing
    "anomaly_cols": [                # weather features only - NOT rainfall itself
        "T2M", "RH2M", "PS", "WS2M", "CLOUD_AMT", "ALLSKY_SFC_SW_DWN", "T2MDEW",
    ],
}


def load_dataset(path: str | Path | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Load the processed rainfall dataset and derive the fields the EDA
    scripts need but the raw preprocessing output doesn't contain:
    Region (for chart-readable grouping) and Rainy_Day (binary proxy
    target, since rainfall itself is continuous).
    """
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")

    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])

    df["Region"] = df["Station"].map(REGION_MAP)
    unmapped = df.loc[df["Region"].isna(), "Station"].unique()
    if len(unmapped) > 0:
        raise ValueError(f"Stations missing from REGION_MAP: {list(unmapped)}")

    df["Rainy_Day"] = df["BMD_Rainfall"] > 0

    return df, CFG


def output_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    return OUTPUT_ROOT