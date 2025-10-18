
"""
visualizations.py
-----------------
Visualization components for the Predictive Maintenance dashboard.

Builds 4 visualization functions:

1. plot_health_trend(df, machine_id, days=7, lang=None)
   - Shows health scores for the last N days (default 7)
   - Bilingual labels: "Health Score" / "درجة الصحة"

2. plot_vibration_analysis(vibration_data, fs_hz=1000, lang=None)
   - Frequency spectrum charts for vibration signals (X/Y/Z if available)
   - Bilingual: "Vibration Analysis" / "تحليل الاهتزازات"

3. plot_failure_gauges(predictions, lang=None)
   - Gauge charts for each failure probability
   - Translated: "Bearing Wear" / "تآكل المحامل", etc.

4. plot_alert_timeline(alerts_df, lang=None)
   - Timeline of alert history with localized dates (basic localization for ar/en).
"""

from __future__ import annotations
import math
from typing import Dict, Iterable, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Optional plotting backends
try:
    import plotly.graph_objects as go
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

import matplotlib.pyplot as plt

# ---------------------------- WADI COLOR PALETTE ----------------------------
WADI_COLORS = {
    "critical": "#051222",    # Dark Blue - High risk
    "warning": "#3DA6DE",     # Light Blue - Medium risk  
    "excellent": "#4EBC83",   # Green - Good health
    "good": "#4EBC83",        # Green - Normal operation
    "background": "#C8ECFC",  # Light Blue - Backgrounds
    "info": "#C8ECFC"         # Light Blue - Info items
}


# Try to use Streamlit translations if available
try:
    import streamlit as st
    from locales.language_manager import get_text
except Exception:
    st = None
    get_text = None


# ---------------------------- i18n helpers ----------------------------

_AR_NUMS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

_DEFAULT_STRINGS = {
    "en": {
        "health_score": "Health Score",
        "vibration_analysis": "Vibration Analysis",
        "bearing_wear": "Bearing Wear",
        "imbalance": "Imbalance",
        "misalignment": "Misalignment",
        "normal": "Normal",
        "alerts": "Alerts",
        "time": "Time",
        "probability": "Probability",
        "alert_timeline": "Alert Timeline",
        "score": "Score",
    },
    "ar": {
        "health_score": "درجة الصحة",
        "vibration_analysis": "تحليل الاهتزازات",
        "bearing_wear": "تآكل المحامل",
        "imbalance": "اختلال الاتزان",
        "misalignment": "عدم المحاذاة",
        "normal": "طبيعي",
        "alerts": "التنبيهات",
        "time": "الوقت",
        "probability": "الاحتمال",
        "alert_timeline": "سجل التنبيهات",
        "score": "الدرجة",
    },
}

def _current_lang(lang: Optional[str]) -> str:
    if lang in ("en", "ar"):
        return lang
    if st is not None and "language" in getattr(st.session_state, "keys", lambda: {})():
        return st.session_state.language
    return "en"

def _t(key: str, lang: Optional[str] = None) -> str:
    # Prefer Streamlit's session translations if available
    if get_text is not None:
        try:
            return get_text(key)
        except Exception:
            pass
    lang = _current_lang(lang)
    return _DEFAULT_STRINGS.get(lang, _DEFAULT_STRINGS["en"]).get(key, key)

def _fmt_time(ts: pd.Timestamp, lang: Optional[str]) -> str:
    # Simple localization: Arabic digits for 'ar' and DD/MM HH:MM format
    lang = _current_lang(lang)
    s = ts.strftime("%Y-%m-%d %H:%M")
    if lang == "ar":
        # convert digits to Arabic-Indic
        s = s.translate(_AR_NUMS)
    return s


# ---------------------------- health trend ----------------------------

def plot_health_trend(df: pd.DataFrame, machine_id: str, days: int = 7, lang: Optional[str] = None):
    """
    Plot health scores over the last `days` days for a specific machine.
    Requires the dataframe to already contain the processed rolling features that
    `health_scorer.compute_health_scores` expects.
    """
    from health_scorer import compute_health_scores  # local import to avoid hard dep for non-users

    dff = df.copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    dff = dff.sort_values("timestamp")
    dff = dff[dff["machine_id"] == machine_id]

    if dff.empty:
        raise ValueError(f"No rows for machine_id={machine_id}")

    # Keep only last N days
    end = dff["timestamp"].max()
    start = end - pd.Timedelta(days=days)
    dff = dff[dff["timestamp"].between(start, end)]

    scores = compute_health_scores(dff)
    dff = dff.loc[scores.index]
    title_text = f"{_t('health_score', lang)} — {machine_id}"

    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dff["timestamp"],
            y=scores,
            mode="lines+markers",
            name=_t("score", lang),
        ))
        fig.update_layout(
            title=title_text,
            xaxis_title=_t("time", lang),
            yaxis_title=_t("health_score", lang),
            yaxis=dict(range=[0, 100]),
            template="plotly_white",
            height=400,
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(dff["timestamp"], scores, marker="o")
        ax.set_title(title_text)
        ax.set_xlabel(_t("time", lang))
        ax.set_ylabel(_t("health_score", lang))
        ax.set_ylim(0, 100)
        fig.autofmt_xdate()
        return fig


# ------------------------ vibration analysis (FFT) ------------------------

def _fft_mag(x: np.ndarray, fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= 1:
        return np.array([]), np.array([])
    X = np.fft.rfft(x - np.nanmean(x))
    freqs = np.fft.rfftfreq(n, d=1.0/fs_hz)
    mag = np.abs(X) * (2.0 / n)  # scale
    return freqs, mag

def plot_vibration_analysis(vibration_data: Union[pd.DataFrame, Dict[str, Iterable], pd.Series, np.ndarray],
                            fs_hz: float = 1000.0,
                            lang: Optional[str] = None):
    """
    Plot frequency spectrum for vibration signals.
    Accepts:
      - DataFrame with columns: vib_x, vib_y, vib_z (any subset)
      - Dict like {"vib_x": array, "vib_y": array, ...}
      - Single array/Series (treated as one axis "vib_x")
    """
    if isinstance(vibration_data, pd.DataFrame):
        cols = [c for c in ["vib_x", "vib_y", "vib_z"] if c in vibration_data.columns]
        data = {c: np.asarray(vibration_data[c].values, dtype=float) for c in cols}
    elif isinstance(vibration_data, dict):
        data = {k: np.asarray(v, dtype=float) for k, v in vibration_data.items() if k in ("vib_x","vib_y","vib_z")}
    else:
        data = {"vib_x": np.asarray(vibration_data, dtype=float)}

    if not data:
        raise ValueError("No vibration data provided. Expect columns or keys: vib_x/vib_y/vib_z")

    spectra = {}
    for axis, arr in data.items():
        arr = np.nan_to_num(arr, nan=np.nanmean(arr) if np.isfinite(np.nanmean(arr)) else 0.0)
        f, m = _fft_mag(arr, fs_hz=fs_hz)
        spectra[axis] = (f, m)

    title_text = _t("vibration_analysis", lang)

    if _HAS_PLOTLY:
        fig = go.Figure()
        for axis, (f, m) in spectra.items():
            fig.add_trace(go.Scatter(x=f, y=m, mode="lines", name=axis.upper()))
        fig.update_layout(
            title=title_text,
            xaxis_title="Hz",
            yaxis_title="Amplitude",
            template="plotly_white",
            height=420,
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for axis, (f, m) in spectra.items():
            ax.plot(f, m, label=axis.upper())
        ax.set_title(title_text)
        ax.set_xlabel("Hz")
        ax.set_ylabel("Amplitude")
        ax.legend()
        return fig


# ------------------------ failure probability gauges ------------------------

_GAUGE_LABELS = {
    "normal": ("normal", "طبيعي"),
    "bearing_wear": ("bearing_wear", "تآكل المحامل"),
    "imbalance": ("imbalance", "اختلال الاتزان"),
    "misalignment": ("misalignment", "عدم المحاذاة"),
}

def plot_failure_gauges(predictions: Union[Dict[str, float], pd.Series], lang: Optional[str] = None):
    """
    Draw gauge indicators for each failure probability.
    `predictions` is expected to map class -> probability in [0,1].
    """
    # Normalize input
    if isinstance(predictions, pd.Series):
        probs = predictions.to_dict()
    else:
        probs = dict(predictions)

    # Ensure all keys present
    for k in ("normal", "bearing_wear", "imbalance", "misalignment"):
        probs.setdefault(k, 0.0)

    # Prepare localized labels
    lang = _current_lang(lang)
    def label_for(key):
        if lang == "ar":
            return _GAUGE_LABELS[key][1]
        # fall back to key→default mapping using our _t when possible
        map_en = {
            "normal": _t("normal", lang),
            "bearing_wear": _t("bearing_wear", lang),
            "imbalance": _t("imbalance", lang),
            "misalignment": _t("misalignment", lang),
        }
        return map_en[key]

    if _HAS_PLOTLY:
        rows = 1; cols = 4
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=rows, cols=cols, specs=[[{"type":"indicator"}]*cols])
        keys = ["normal","bearing_wear","imbalance","misalignment"]
        for i, k in enumerate(keys, start=1):
            val = float(probs.get(k, 0.0)) * 100.0
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=val,
                    title={"text": label_for(k)},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"thickness": 0.5},
                        "steps": [
                            {"range": [0, 33], "color": "WADI_COLORS["critical"]"},
                            {"range": [33, 66], "color": "WADI_COLORS["warning"]"},
                            {"range": [66, 100], "color": "WADI_COLORS["excellent"]"},
                        ],
                    },
                    number={"suffix": "%"}
                ),
                row=1, col=i
            )
        fig.update_layout(height=300, margin=dict(t=40, l=20, r=20, b=20))
        return fig
    else:
        # Matplotlib fallback: simple bar gauges
        keys = ["normal","bearing_wear","imbalance","misalignment"]
        vals = [float(probs.get(k,0.0))*100 for k in keys]
        labels = [label_for(k) for k in keys]
        fig, ax = plt.subplots(figsize=(8, 2.8))
        bars = ax.bar(np.arange(len(keys)), vals)
        ax.set_xticks(np.arange(len(keys)), labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel(_t("probability", lang))
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
        return fig


# ------------------------ alert timeline ------------------------

def plot_alert_timeline(alerts_df: pd.DataFrame, lang: Optional[str] = None):
    """
    Plot a timeline of alerts.
    Expected columns:
      - 'timestamp': datetime
      - 'level': string (e.g., 'Critical', 'Warning', 'Info')
      - 'message': string
    """
    df = alerts_df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("alerts_df must have a 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Color map
    cmap = {
        "Critical": "WADI_COLORS["critical"]",
        "Warning": "WADI_COLORS["warning"]",
        "Good (monitor)": "WADI_COLORS["good"]",
        "Excellent": "WADI_COLORS["excellent"]",
        "Info": "WADI_COLORS["info"]",
        "Normal": "WADI_COLORS["info"]",
    }
    colors = [cmap.get(str(x), "#888") for x in df.get("level", pd.Series(["Info"]*len(df)))]

    title_text = _t("alert_timeline", lang)

    if _HAS_PLOTLY:
        # Scatter timeline
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=[1]*len(df),
            mode="markers+text",
            marker=dict(size=12, color=colors),
            text=[str(m) for m in df.get("message", pd.Series([""]*len(df)))],
            textposition="top center",
            hovertext=[_fmt_time(ts, lang) for ts in df["timestamp"]],
            hoverinfo="text",
        ))
        fig.update_yaxes(visible=False, range=[0.8, 1.2])
        fig.update_layout(
            title=title_text,
            xaxis_title=_t("time", lang),
            template="plotly_white",
            height=280,
            margin=dict(t=50, l=30, r=30, b=40),
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.scatter(df["timestamp"], [1]*len(df), c=colors, s=40)
        for ts, msg in zip(df["timestamp"], df.get("message", pd.Series([""]*len(df)))):
            ax.text(ts, 1.02, str(msg), rotation=45, ha="left", va="bottom", fontsize=8)
        ax.set_yticks([])
        ax.set_title(title_text)
        ax.set_xlabel(_t("time", lang))
        fig.autofmt_xdate()
        return fig
