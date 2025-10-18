# locales/language_manager.py
import json
from pathlib import Path
import streamlit as st

BASE = Path(__file__).parent  # always points to the locales/ folder

def load_translations(lang: str):
    """Load translations from locales/<lang>.json."""
    with open(BASE / f"{lang}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_text(key: str):
    """Read a translated string from st.session_state.translations; fallback to the key."""
    return st.session_state.get("translations", {}).get(key, key)

def initialize_language(default_lang: str = "en"):
    """Initialize language + translations in Streamlit session_state."""
    if "language" not in st.session_state:
        st.session_state.language = default_lang
    if "translations" not in st.session_state:
        st.session_state.translations = load_translations(st.session_state.language)
