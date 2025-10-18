# app.py
# ------------------------------
# Wadi Data — Predictive Maintenance Dashboard
# ------------------------------

from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from locales.language_manager import initialize_language, load_translations, get_text
from visualizations import (
    plot_health_trend,
    plot_vibration_analysis,
    plot_failure_gauges,
    plot_alert_timeline,
    _current_lang,
    _t,
)
from health_scorer import compute_health_scores, classify_band
from failure_predictor import train_model, load_model, predict_proba_df
from data_generator import generate_dataset
from data_processor import clean_sensor_data, add_rolling_stats, add_vibration_features

# ---------------------------- WADI BRAND ----------------------------
WADI_COLORS = {
    "critical": "#051222",
    "warning": "#3DA6DE",
    "excellent": "#4EBC83",
    "good": "#4EBC83",
    "background": "#C8ECFC",
    "info": "#C8ECFC"
}

def wadi_badge(text: str):
    html = (
        "<div style=\"background:{bg};color:{fg};padding:8px 12px;"
        "border-radius:12px;display:inline-block;font-weight:600;\">"
        "{text}</div>"
    ).format(bg=WADI_COLORS['background'], fg=WADI_COLORS['critical'], text=text)
    st.markdown(html, unsafe_allow_html=True)

def wadi_title(text: str):
    html = "<h1 style='color:{fg};margin-bottom:0.2rem;'>{t}</h1>".format(fg=WADI_COLORS['critical'], t=text)
    st.markdown(html, unsafe_allow_html=True)

def wadi_subtitle(text: str):
    html = "<h3 style='color:{fg};margin-top:0.25rem;'>{t}</h3>".format(fg=WADI_COLORS['critical'], t=text)
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------- SESSION / LANGUAGE ----------------------------

def set_language_ui():
    initialize_language()
    lang = st.sidebar.selectbox("🌐 Language / اللغة", ["en", "ar"],
                                index=0 if st.session_state.language == "en" else 1)
    if lang != st.session_state.language:
        st.session_state.language = lang
        st.session_state.translations = load_translations(lang)
    return lang

# ---------------------------- DATA SOURCING ----------------------------

@st.cache_data(show_spinner=False)
def load_or_generate(days:int=7, n_machines:int=6, freq:str="2min", seed:int=1337):
    df, meta = generate_dataset(days=days, n_machines=n_machines, freq=freq, seed=seed)
    df_clean = clean_sensor_data(df, resample_freq=freq)
    df_roll  = add_rolling_stats(df_clean, window="60min")
    df_feat  = add_vibration_features(df_roll, window="60min")
    return df_feat, meta

@st.cache_resource(show_spinner=False)
def get_model(model_path:str="rf_failure_model.joblib"):
    try:
        bundle = load_model(model_path)
        return bundle
    except Exception:
        res = train_model(train_csv=None, model_out=model_path)
        return load_model(model_path)

# ---------------------------- TRANSLATION SHORTCUTS ----------------------------

def T(key: str) -> str:
    try:
        return get_text(key)
    except Exception:
        return _t(key, _current_lang(None))

# ---------------------------- NAVIGATION ----------------------------

PAGES = {
    "overview": {"en": "Overview", "ar": "نظرة عامة"},
    "health": {"en": "Equipment Health", "ar": "صحة المعدات"},
    "vibration": {"en": "Vibration", "ar": "الاهتزازات"},
    "predictions": {"en": "Predictions", "ar": "التنبؤات"},
    "alerts": {"en": "Alerts", "ar": "التنبيهات"},
    "settings": {"en": "Settings", "ar": "الإعدادات"},
}

def page_label(key: str, lang: str) -> str:
    return PAGES.get(key, {}).get(lang, key)

# ---------------------------- SIDEBAR ----------------------------

def sidebar_controls(df: pd.DataFrame, lang: str):
    # Logo section
    st.sidebar.markdown("""
    <div style="text-align:center; padding:10px; border-bottom: 2px solid #3DA6DE; margin-bottom: 20px;">
        <h3 style="color:#051222; margin:0; font-weight:bold;">WADI DATA</h3>
        <p style="color:#3DA6DE; margin:0; font-size:14px;">Predictive Maintenance</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📂 " + ("Navigation" if lang=="en" else "التنقل"),
        list(PAGES.keys()),
        format_func=lambda k: page_label(k, lang)
    )
    machines = sorted(df["machine_id"].unique())
    machine = st.sidebar.selectbox(("Machine" if lang=="en" else "الآلة"), machines)
    days = st.sidebar.slider(("Last days" if lang=="en" else "آخر أيام"), 1, 30, 7)
    return page, machine, days

# ---------------------------- MAIN PAGES ----------------------------

def page_overview(df: pd.DataFrame, meta: dict, lang: str):
    wadi_title(T("dashboard_title"))
    total_machines = df["machine_id"].nunique()
    last_ts = pd.to_datetime(df["timestamp"]).max()
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        wadi_badge(("🛠️ Machines: {n}").format(n=total_machines) if lang=="en" else "🛠️ عدد الآلات: {n}".format(n=total_machines))
    with c2:
        wadi_badge(("⏱️ Last Update: {t}").format(t=last_ts.strftime('%Y-%m-%d %H:%M')) if lang=="en" else "⏱️ آخر تحديث: {t}".format(t=last_ts.strftime('%Y-%m-%d %H:%M')))
    with c3:
        wadi_badge("Wadi Data • PdM")

    st.markdown("---")
    st.subheader(page_label("health", lang))
    scores = compute_health_scores(df)
    dfh = pd.DataFrame({
        "machine_id": df.loc[scores.index, "machine_id"].values,
        "timestamp": pd.to_datetime(df.loc[scores.index, "timestamp"].values),
        "health": scores.values
    })
    latest = dfh.sort_values("timestamp").groupby("machine_id").tail(1).sort_values("health", ascending=False)
    st.dataframe(latest[["machine_id","health"]].set_index("machine_id", use_container_width=True))

def page_health(df: pd.DataFrame, machine_id: str, days: int, lang: str):
    wadi_subtitle(page_label("health", lang) + " — " + machine_id)
    fig = plot_health_trend(df, machine_id, days=days, lang=lang)
    if hasattr(fig, "to_dict"):
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, use_container_width=True)

def page_vibration(df: pd.DataFrame, machine_id: str, days: int, lang: str):
    wadi_subtitle(page_label("vibration", lang) + " — " + machine_id)
    dff = df[df["machine_id"] == machine_id].copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    dff = dff.sort_values("timestamp").tail(2048)
    vib = dff[["vib_x","vib_y","vib_z"]].dropna().tail(2048)
    fig = plot_vibration_analysis(vib, fs_hz=1000.0, lang=lang)
    if hasattr(fig, "to_dict"):
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, use_container_width=True)

def page_predictions(df: pd.DataFrame, machine_id: str, days: int, lang: str):
    wadi_subtitle(page_label("predictions", lang) + " — " + machine_id)
    model = get_model()
    dff = df[df["machine_id"] == machine_id].copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    dff = dff.sort_values("timestamp").tail(500)
    probs_df = predict_proba_df(model, dff).tail(1)
    if probs_df.empty:
        st.warning("No predictions were generated for this selection." if lang=="en" else "لم يتم توليد تنبؤات لهذا الاختيار.")
        return
    probs = probs_df.iloc[0].to_dict()
    fig = plot_failure_gauges(probs, lang=lang)
    if hasattr(fig, "to_dict"):
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, use_container_width=True)
    st.write(pd.DataFrame([probs]).T.rename(columns={0:"probability"}))

def page_alerts(df: pd.DataFrame, machine_id: str, days: int, lang: str):
    wadi_subtitle(page_label("alerts", lang) + " — " + machine_id)
    dff = df[df["machine_id"] == machine_id].copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    dff = dff.sort_values("timestamp")
    scores = compute_health_scores(dff)
    bands = classify_band(scores)
    end = dff["timestamp"].max(); start = end - pd.Timedelta(days=days)
    mask = dff["timestamp"].between(start, end)
    alerts_df = pd.DataFrame({
        "timestamp": dff.loc[mask, "timestamp"],
        "level": bands.loc[mask].astype(str),
        "message": (["Check vibration levels"]*int(mask.sum())) if lang=="en" else (["تحقق من مستويات الاهتزاز"]*int(mask.sum()))
    })
    fig = plot_alert_timeline(alerts_df, lang=lang)
    if hasattr(fig, "to_dict"):
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig, use_container_width=True)
    st.dataframe(alerts_df, use_container_width=True)

def page_settings(df: pd.DataFrame, meta: dict, lang: str):
    wadi_subtitle(page_label("settings", lang))
    st.write("Upload a CSV to replace the synthetic data" if lang=="en" else "حمّل ملف CSV لاستبدال البيانات الاصطناعية")
    up = st.file_uploader("CSV", type=["csv"])
    if up is not None:
        df_new = pd.read_csv(up)
        st.session_state["data_df"] = df_new

    st.markdown("---")
    if st.button("Re-train model" if lang=="en" else "إعادة تدريب النموذج"):
        with st.spinner("Training model..." if lang=="en" else "جارٍ تدريب النموذج..."):
            res = train_model(train_csv=None, model_out="rf_failure_model.joblib")
            st.success("Accuracy: {:.3f}".format(res['accuracy']))
            st.text(res["classification_report"])

# ---------------------------- APP ENTRY ----------------------------

def main():
    st.set_page_config(page_title="Wadi Data • PdM", page_icon="🏭", layout="wide")
    lang = set_language_ui()

    if "data_df" not in st.session_state:
        df, meta = load_or_generate(days=7, n_machines=6, freq="2min", seed=1337)
        st.session_state["data_df"] = df
        st.session_state["data_meta"] = meta
    df = st.session_state["data_df"]
    meta = st.session_state.get("data_meta", {})

    page, machine, days = sidebar_controls(df, lang)

    if page == "overview":
        page_overview(df, meta, lang)
    elif page == "health":
        page_health(df, machine, days, lang)
    elif page == "vibration":
        page_vibration(df, machine, days, lang)
    elif page == "predictions":
        page_predictions(df, machine, days, lang)
    elif page == "alerts":
        page_alerts(df, machine, days, lang)
    elif page == "settings":
        page_settings(df, meta, lang)

if __name__ == "__main__":
    main()
