# locales/language_manager.py
import json
from pathlib import Path
import streamlit as st

BASE = Path(__file__).parent  # → the locales/ folder regardless of CWD

def load_translations(lang: str):
    with open(BASE / f"{lang}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_text(key: str):
    return st.session_state.translations.get(key, key)

def initialize_language():
    if "language" not in st.session_state:
        st.session_state.language = "en"
        st.session_state.translations = load_translations("en")
