
"""
health_scorer.py
----------------
Composite health scoring (0-100) from multiple sensors.

Bands:
- 90-100: Excellent
- 70-89: Good (monitor)
- 50-69: Warning
- 0-49:  Critical

Usage (API):
    from health_scorer import compute_health_scores, classify_band
    scores = compute_health_scores(df)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, x))


def compute_health_scores(df: pd.DataFrame) -> pd.Series:
    """
    Compute a 0-100 health score based on available columns.
    Expected columns (best-effort; missing columns fall back to safer defaults):
      - vib_x_rms_60min, vib_y_rms_60min, vib_z_rms_60min
      - vib_x_kurtosis_60min, vib_y_kurtosis_60min, vib_z_kurtosis_60min
      - vib_x_crest_60min, vib_y_crest_60min, vib_z_crest_60min
      - temperature, pressure, rpm_std_60min
    """
    def get(col, default=0.0):
        return df[col].values if col in df.columns else np.full(len(df), default)

    vrms = np.nanmean(
        np.vstack([get("vib_x_rms_60min"), get("vib_y_rms_60min"), get("vib_z_rms_60min")]), axis=0
    )
    rms_risk = _clip01((vrms - 3.0) / (10.0 - 3.0))

    vkurt = np.nanmean(
        np.vstack([get("vib_x_kurtosis_60min"), get("vib_y_kurtosis_60min"), get("vib_z_kurtosis_60min")]), axis=0
    )
    kurt_risk = _clip01((vkurt - 3.0) / (6.0 - 3.0))

    vcrest = np.nanmean(
        np.vstack([get("vib_x_crest_60min"), get("vib_y_crest_60min"), get("vib_z_crest_60min")]), axis=0
    )
    crest_risk = _clip01((vcrest - 2.0) / (5.0 - 2.0))

    temp = get("temperature")
    temp_risk = _clip01((temp - 35.0) / (70.0 - 35.0))

    pressure = get("pressure")
    pressure_risk = _clip01(np.abs(pressure - 8.0) / 2.0)

    rpm_std = get("rpm_std_60min")
    rpm_var_risk = _clip01(rpm_std / 60.0)

    risk = (
        0.30 * rms_risk +
        0.20 * kurt_risk +
        0.15 * crest_risk +
        0.15 * temp_risk +
        0.10 * pressure_risk +
        0.10 * rpm_var_risk
    )

    health = (1.0 - _clip01(risk)) * 100.0
    return pd.Series(health, index=df.index).round(1)


def classify_band(scores: pd.Series) -> pd.Series:
    return pd.cut(
        scores,
        bins=[-0.1, 49, 69, 89, 100.1],
        labels=["Critical", "Warning", "Good (monitor)", "Excellent"]
    )
