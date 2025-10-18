
"""
alert_system.py
----------------
Alerting utilities for the Wadi Data Predictive Maintenance dashboard.

Updated to support:
- Triggers:
    * Health score < 70  → High
    * Health score < 50  → Critical
    * Vibration spikes   → Medium
    * Temperature anomalies → Low
- Session-state storage for Streamlit (acknowledge/resolve/filter/clear)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Iterable, Literal, Tuple
import uuid
import time
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# Priority system (with brand colors and Arabic labels)
# ------------------------------------------------------------------
PRIORITY_LEVELS: Dict[str, Dict[str, str]] = {
    "low":      {"en": "Low",      "ar": "منخفض",  "color": "#3DA6DE"},
    "medium":   {"en": "Medium",   "ar": "متوسط",  "color": "#3DA6DE"},   # using WADI light blue
    "high":     {"en": "High",     "ar": "مرتفع",  "color": "#3DA6DE"},   # stylistic: can map to another if desired
    "critical": {"en": "Critical", "ar": "حرج",    "color": "#051222"}
}

# Optional integration: use Streamlit translations if available
try:
    from locales.language_manager import get_text
except Exception:
    get_text = None


# ------------------------------------------------------------------
# Triggers config
# ------------------------------------------------------------------
HEALTH_HIGH_THRESHOLD = 70.0     # <70 => High
HEALTH_CRIT_THRESHOLD = 50.0     # <50 => Critical

VIB_SPIKE_Z = 3.0                # z-score threshold for spikes (any axis)
TEMP_ANOM_Z = 2.5                # z-score threshold for temperature anomalies
TEMP_ABS_LIMIT = 70.0            # °C absolute limit as anomaly


# ------------------------------------------------------------------
# Detection helpers
# ------------------------------------------------------------------
def _rolling_zscore(x: pd.Series, window: str | int = "120min", min_periods: int = 20) -> pd.Series:
    """Time-based or length-based rolling z-score."""
    roll = x.rolling(window, min_periods=min_periods)
    mu = roll.mean()
    sd = roll.std().replace(0, np.nan)
    return (x - mu) / sd

def detect_vibration_spikes(df: pd.DataFrame) -> pd.Series:
    """
    Return boolean Series indicating vibration spikes (any axis exceeding z-threshold).
    Requires columns 'timestamp', 'machine_id' and at least one of 'vib_x','vib_y','vib_z'.
    Works per machine with a 120min rolling z-score.
    """
    if "timestamp" not in df.columns or "machine_id" not in df.columns:
        return pd.Series(False, index=df.index)

    axes = [c for c in ["vib_x","vib_y","vib_z"] if c in df.columns]
    if not axes:
        return pd.Series(False, index=df.index)

    out = pd.Series(False, index=df.index)
    dff = df.copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    dff = dff.sort_values(["machine_id","timestamp"])
    for mid, g in dff.groupby("machine_id", sort=False):
        spikes_any = pd.Series(False, index=g.index)
        for ax in axes:
            z = _rolling_zscore(g[ax])
            spikes_any = spikes_any | (z.abs() >= VIB_SPIKE_Z)
        out.loc[g.index] = spikes_any
    return out.reindex(df.index).fillna(False)

def detect_temperature_anomalies(df: pd.DataFrame) -> pd.Series:
    """
    Return boolean Series indicating temperature anomalies (rolling z > TEMP_ANOM_Z or absolute exceedance).
    Requires 'temperature', 'timestamp', 'machine_id'.
    """
    req = {"temperature","timestamp","machine_id"}
    if not req.issubset(df.columns):
        return pd.Series(False, index=df.index)

    out = pd.Series(False, index=df.index)
    dff = df.copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    dff = dff.sort_values(["machine_id","timestamp"])
    for mid, g in dff.groupby("machine_id", sort=False):
        z = _rolling_zscore(g["temperature"])
        bools = (z.abs() >= TEMP_ANOM_Z) | (g["temperature"] >= TEMP_ABS_LIMIT)
        out.loc[g.index] = bools
    return out.reindex(df.index).fillna(False)


# ------------------------------------------------------------------
# Priority decisions
# ------------------------------------------------------------------
def priority_for_metrics(health: float, vib_spike: bool, temp_anom: bool) -> str:
    """
    Apply trigger precedence:
      1) Critical if health < 50
      2) High     if health < 70
      3) Medium   if vibration spikes present
      4) Low      if temperature anomaly present
      else 'none'
    """
    if health < HEALTH_CRIT_THRESHOLD:
        return "critical"
    if health < HEALTH_HIGH_THRESHOLD:
        return "high"
    if vib_spike:
        return "medium"
    if temp_anom:
        return "low"
    return "none"


# ------------------------------------------------------------------
# Alert data model
# ------------------------------------------------------------------
@dataclass
class Alert:
    machine_id: str
    timestamp: pd.Timestamp
    health_score: float
    priority: Literal["low","medium","high","critical"]
    title_en: str
    title_ar: str
    message_en: str
    message_ar: str
    color: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    acknowledged: bool = False
    resolved: bool = False
    channel_log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["timestamp"] = pd.to_datetime(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return d


# ------------------------------------------------------------------
# Message builder
# ------------------------------------------------------------------
def _msg_parts(machine_id: str, score: float, pr: str) -> Dict[str, str]:
    en_title = f"Health alert for {machine_id}"
    ar_title = f"تنبيه صحة الآلة {machine_id}"
    if pr == "critical":
        en_msg = f"Critical: health score {score:.1f}. Immediate action required."
        ar_msg = f"حرج: درجة الصحة {score:.1f}. مطلوب تدخل فوري."
    elif pr == "high":
        en_msg = f"High priority: health score {score:.1f}. Please investigate soon."
        ar_msg = f"أولوية مرتفعة: درجة الصحة {score:.1f}. يرجى التحقق قريبًا."
    elif pr == "medium":
        en_msg = "Vibration spike detected. Monitor equipment condition."
        ar_msg = "تم رصد ارتفاع مفاجئ في الاهتزاز. يرجى المراقبة."
    elif pr == "low":
        en_msg = "Temperature anomaly detected. Monitor trends."
        ar_msg = "تم رصد شذوذ في درجة الحرارة. يرجى المراقبة."
    else:
        en_msg = f"Score {score:.1f}."
        ar_msg = f"الدرجة {score:.1f}."
    return {
        "title_en": en_title,
        "title_ar": ar_title,
        "message_en": en_msg,
        "message_ar": ar_msg,
    }


# ------------------------------------------------------------------
# Alert generation from metrics
# ------------------------------------------------------------------
def generate_alerts(df: pd.DataFrame) -> List[Alert]:
    """
    Generate alerts from a dataframe with at least:
      - 'timestamp', 'machine_id'
      - 'health' (0..100)
      - optionally vibration columns ('vib_x','vib_y','vib_z') and 'temperature' for extra triggers
    """
    required = {"timestamp","machine_id","health"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    vib_bool = detect_vibration_spikes(df) if any(c in df.columns for c in ["vib_x","vib_y","vib_z"]) else pd.Series(False, index=df.index)
    temp_bool = detect_temperature_anomalies(df) if "temperature" in df.columns else pd.Series(False, index=df.index)

    alerts: List[Alert] = []
    for idx, row in df.iterrows():
        health = float(row["health"])
        pr = priority_for_metrics(health, bool(vib_bool.loc[idx]), bool(temp_bool.loc[idx]))
        if pr == "none":
            continue
        parts = _msg_parts(str(row["machine_id"]), health, pr)
        info = PRIORITY_LEVELS[pr]
        alerts.append(Alert(
            machine_id=str(row["machine_id"]),
            timestamp=pd.to_datetime(row["timestamp"]),
            health_score=health,
            priority=pr,
            title_en=parts["title_en"],
            title_ar=parts["title_ar"],
            message_en=parts["message_en"],
            message_ar=parts["message_ar"],
            color=info["color"],
        ))
    return alerts


# ------------------------------------------------------------------
# Generate from raw (compute health if needed)
# ------------------------------------------------------------------
def generate_from_raw(df: pd.DataFrame) -> List[Alert]:
    """
    If 'health' not present, compute health scores using health_scorer (src.* fallback to root).
    Then call generate_alerts.
    """
    if "health" not in df.columns:
        try:
            from src.health_scorer import compute_health_scores
        except Exception:
            from health_scorer import compute_health_scores
        scores = compute_health_scores(df)
        df = df.loc[scores.index].copy()
        df["health"] = scores.values
    return generate_alerts(df)


# ------------------------------------------------------------------
# AlertManager: in-memory store + session-state helpers
# ------------------------------------------------------------------
class AlertManager:
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}

    def add(self, alert: Alert) -> str:
        self._alerts[alert.id] = alert
        return alert.id

    def add_many(self, alerts: Iterable[Alert]) -> List[str]:
        return [self.add(a) for a in alerts]

    def list(self, include_resolved: bool = True) -> pd.DataFrame:
        rows = [a.to_dict() for a in self._alerts.values() if include_resolved or not a.resolved]
        if not rows:
            return pd.DataFrame(columns=["id","timestamp","machine_id","priority","health_score","acknowledged","resolved","title_en","message_en","title_ar","message_ar","color","channel_log"])
        return pd.DataFrame(rows).sort_values("timestamp")

    def get(self, alert_id: str) -> Optional[Alert]:
        return self._alerts.get(alert_id)

    def acknowledge(self, alert_id: str) -> bool:
        a = self._alerts.get(alert_id)
        if not a: 
            return False
        a.acknowledged = True
        return True

    def resolve(self, alert_id: str) -> bool:
        a = self._alerts.get(alert_id)
        if not a:
            return False
        a.resolved = True
        return True

    def clear_before(self, before_ts: pd.Timestamp) -> int:
        """Remove alerts older than before_ts. Returns count removed."""
        ids = [k for k, a in self._alerts.items() if pd.to_datetime(a.timestamp) < pd.to_datetime(before_ts)]
        for k in ids:
            del self._alerts[k]
        return len(ids)

    # ---- Notification simulation ----
    def _log_channel(self, alert: Alert, channel: str, payload: str):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        alert.channel_log.append(f"{channel}@{stamp}: {payload}")

    def notify_email(self, alert_id: str, to: str) -> bool:
        a = self._alerts.get(alert_id)
        if not a:
            return False
        subject = f"[{PRIORITY_LEVELS[a.priority]['en']}] {a.title_en}"
        body = f"{a.message_en}\nMachine: {a.machine_id}\nPriority: {a.priority}\nScore: {a.health_score:.1f}"
        self._log_channel(a, "email", f"to={to} | subject={subject} | body={body}")
        return True

    def notify_sms(self, alert_id: str, to: str) -> bool:
        a = self._alerts.get(alert_id)
        if not a:
            return False
        msg = f"[{PRIORITY_LEVELS[a.priority]['en']}] {a.machine_id}: {a.message_en} (Score {a.health_score:.1f})"
        self._log_channel(a, "sms", f"to={to} | msg={msg}")
        return True


# ------------------------------------------------------------------
# Streamlit helpers: session-state store & filtering
# ------------------------------------------------------------------
def ensure_session_manager() -> AlertManager:
    """Get or create an AlertManager in st.session_state['alert_manager']"""
    try:
        import streamlit as st
    except Exception:
        # outside streamlit: return a plain manager
        return AlertManager()
    if "alert_manager" not in st.session_state:
        st.session_state["alert_manager"] = AlertManager()
    return st.session_state["alert_manager"]

def add_alerts_to_session(alerts: List[Alert]) -> None:
    am = ensure_session_manager()
    am.add_many(alerts)

def set_alert_status(alert_id: str, acknowledged: Optional[bool] = None, resolved: Optional[bool] = None) -> None:
    am = ensure_session_manager()
    if acknowledged is True:
        am.acknowledge(alert_id)
    if resolved is True:
        am.resolve(alert_id)

def filter_alerts(machine: Optional[str] = None, priority: Optional[str] = None, status: Optional[str] = None) -> pd.DataFrame:
    """
    status ∈ {'all','open','acknowledged','resolved'}
    """
    am = ensure_session_manager()
    df = am.list(include_resolved=True)
    if df.empty:
        return df
    if machine:
        df = df[df["machine_id"] == machine]
    if priority:
        df = df[df["priority"] == priority]
    if status and status != "all":
        if status == "open":
            df = df[df["resolved"] == False]
        elif status == "acknowledged":
            df = df[(df["acknowledged"] == True) & (df["resolved"] == False)]
        elif status == "resolved":
            df = df[df["resolved"] == True]
    return df

def clear_old_alerts(before_ts: pd.Timestamp) -> int:
    am = ensure_session_manager()
    return am.clear_before(before_ts)


# ------------------------------------------------------------------
# Streamlit UI renderers (cards/table)
# ------------------------------------------------------------------
def render_alert_card(a: Alert, lang: str = "en"):
    """
    Render a single alert card using Streamlit with bilingual text and brand color.
    """
    try:
        import streamlit as st
    except Exception:
        return

    title = a.title_en if lang == "en" else a.title_ar
    msg = a.message_en if lang == "en" else a.message_ar
    pr_label = PRIORITY_LEVELS[a.priority]["en"] if lang == "en" else PRIORITY_LEVELS[a.priority]["ar"]
    st.markdown(f"""
    <div style="border-left: 8px solid {a.color}; padding:10px 12px; margin:8px 0; background:#F9FAFB; border-radius:8px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <strong style="font-size:16px;">{title}</strong>
        <span style="background:{a.color}; color:#fff; border-radius:999px; padding:2px 10px; font-size:12px;">{pr_label}</span>
      </div>
      <div style="font-size:13px; margin-top:6px;">{msg}</div>
      <div style="font-size:12px; opacity:0.7; margin-top:6px;">⏱ {pd.to_datetime(a.timestamp).strftime("%Y-%m-%d %H:%M")}</div>
    </div>
    """, unsafe_allow_html=True)

def render_alert_table(am: AlertManager, include_resolved: bool = True):
    try:
        import streamlit as st
    except Exception:
        return
    df = am.list(include_resolved=include_resolved)
    if df.empty:
        st.info("No alerts.")
        return
    st.dataframe(df.set_index("id"))


# ------------------------------------------------------------------
# Self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Build a tiny fake dataset
    n = 12
    ts = pd.date_range("2025-01-01", periods=n, freq="H")
    demo = pd.DataFrame({
        "timestamp": ts,
        "machine_id": ["M01"] * n,
        "health": [92, 85, 74, 69, 62, 55, 48, 35, 72, 68, 66, 90],
        "vib_x": np.r_[np.ones(8)*2.0, np.ones(4)*8.0],   # spikes near end
        "temperature": np.linspace(35, 80, n),            # rising to anomaly
    })
    alerts = generate_alerts(demo)
    am = AlertManager()
    am.add_many(alerts)
    # mark one acknowledge and one resolve
    if alerts:
        am.acknowledge(alerts[0].id)
        am.resolve(alerts[-1].id)
        am.notify_email(alerts[-1].id, "ops@example.com")
        am.notify_sms(alerts[-1].id, "+15551234567")
    print(am.list().to_string(index=False))
