
"""
data_processor.py
-----------------
Data preprocessing pipeline for predictive maintenance.

Functions:
1) clean_sensor_data(df): handle missing values and outliers (per machine)
2) add_rolling_stats(df, window='60min'): rolling mean/std/max over window per machine
3) add_vibration_features(df): RMS, Kurtosis, Crest Factor per-axis and overall

Assumptions:
- Input dataframe has columns:
  ['timestamp','machine_id','rpm','pressure','temperature','vib_x','vib_y','vib_z', ...]
- 'timestamp' is parseable to datetime (or already datetime64[ns])
- Data are (approximately) evenly sampled; operations are time-based using resampling.

Usage:
    from data_processor import clean_sensor_data, add_rolling_stats, add_vibration_features
    df = clean_sensor_data(df)
    df = add_rolling_stats(df, window='60min')
    df = add_vibration_features(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis



def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df['timestamp'])
    # Force timezone-naive timestamps (rollings with strings require naive or consistent tz)
    try:
        ts = ts.dt.tz_convert(None)
    except Exception:
        try:
            ts = ts.dt.tz_localize(None)
        except Exception:
            pass
    df['timestamp'] = ts
    return df
def clean_sensor_data(df: pd.DataFrame, resample_freq: str = "1min", z_clip: float = 4.0) -> pd.DataFrame:
    """
    Per-machine cleaning:
    - Ensure datetime and sort
    - Resample to fixed frequency (forward-fill numeric gaps, limit to short gaps)
    - Interpolate short missing spans (time-based)
    - Winsorize outliers via rolling z-score clipping
    """
    df = _ensure_datetime(df)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cleaned_parts = []
    for mid, g in df.groupby("machine_id", sort=False):
        g = g.set_index("timestamp")

        # uniform sampling
        g = g.resample(resample_freq).asfreq()

        # reattach non-numerics
        g["machine_id"] = mid
        # forward-fill failure_mode/degradation_level if present
        for col in df.columns:
            if col not in g.columns and col in ("failure_mode","degradation_level","failure_event"):
                g[col] = np.nan
        if "failure_mode" in g.columns:
            g["failure_mode"] = g["failure_mode"].ffill().bfill()

        # interpolate numeric columns timewise
        for col in ["rpm","pressure","temperature","vib_x","vib_y","vib_z"]:
            if col in g.columns:
                g[col] = g[col].interpolate(method="time", limit=30, limit_direction="both")

        # rolling z-score clipping for vibration & temp/pressure/rpm
        roll = g[["rpm","pressure","temperature","vib_x","vib_y","vib_z"]].rolling("60min", min_periods=10)
        mu = roll.mean()
        sd = roll.std().replace(0, np.nan)
        z = (g[["rpm","pressure","temperature","vib_x","vib_y","vib_z"]] - mu) / sd
        clipped = g[["rpm","pressure","temperature","vib_x","vib_y","vib_z"]].where(z.abs() <= z_clip, mu + z_clip*sd)
        g[["rpm","pressure","temperature","vib_x","vib_y","vib_z"]] = clipped

        cleaned_parts.append(g.reset_index())

    out = pd.concat(cleaned_parts, ignore_index=True)
    return out


def add_rolling_stats(df: pd.DataFrame, window: str = "60min") -> pd.DataFrame:
    """
    Add rolling mean/std/max for rpm, pressure, temperature, vib_x/y/z per machine.
    """
    df = _ensure_datetime(df)
    df = df.sort_values(["machine_id","timestamp"]).reset_index(drop=True)
    cols = ["rpm","pressure","temperature","vib_x","vib_y","vib_z"]
    parts = []
    for mid, g in df.groupby("machine_id", sort=False):
        g = g.set_index("timestamp")
        roll = g[cols].rolling(window, min_periods=10)
        g[[f"{c}_mean_{window}" for c in cols]] = roll.mean().values
        g[[f"{c}_std_{window}" for c in cols]] = roll.std().values
        g[[f"{c}_max_{window}" for c in cols]] = roll.max().values
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def add_vibration_features(df: pd.DataFrame, window: str = "60min") -> pd.DataFrame:
    """
    Compute rolling vibration features:
    - RMS per axis and overall
    - Kurtosis per axis
    - Crest factor per axis (peak / RMS)
    """
    df = _ensure_datetime(df)
    df = df.sort_values(["machine_id","timestamp"]).reset_index(drop=True)
    cols = ["vib_x","vib_y","vib_z"]
    parts = []
    for mid, g in df.groupby("machine_id", sort=False):
        g = g.set_index("timestamp")
        roll = g[cols].rolling(window, min_periods=10)

        # RMS: sqrt(mean(x^2))
        rms = np.sqrt((roll.apply(lambda x: np.mean(x**2), raw=True)))
        rms.columns = [f"{c}_rms_{window}" for c in cols]

        # Kurtosis: use Fisher=False to get normal=3, then excess kurtosis if needed
        kur = roll.apply(lambda x: kurtosis(x, fisher=False), raw=False)
        kur.columns = [f"{c}_kurtosis_{window}" for c in cols]

        # Crest Factor: peak / RMS over window
        peak = roll.max().abs()
        crest = peak.values / (rms.replace(0, np.nan).values)
        crest_df = pd.DataFrame(crest, index=g.index, columns=[f"{c}_crest_{window}" for c in cols])

        # Overall vibration magnitude (vector)
        g["vib_overall"] = np.sqrt((g["vib_x"]**2 + g["vib_y"]**2 + g["vib_z"]**2))

        g = pd.concat([g, rms, kur, crest_df], axis=1)
        parts.append(g.reset_index())

    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    # Quick demo pipeline (small sample for speed)
    # Generate a tiny dataset if executed standalone.
    try:
        from data_generator import generate_dataset
        df, meta = generate_dataset(days=2, n_machines=2, freq="5min", seed=123)
        df_clean = clean_sensor_data(df, resample_freq="5min")
        df_roll = add_rolling_stats(df_clean, window="60min")
        df_feat = add_vibration_features(df_roll, window="60min")
        df_feat.to_csv("demo_processed.csv", index=False)
        print("Wrote demo_processed.csv")
    except Exception as e:
        print("Demo failed:", e)
