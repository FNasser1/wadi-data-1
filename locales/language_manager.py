import json
import streamlit as st

def load_translations(lang):
    """Load language file (JSON) from locales/ directory."""
    with open(f'locales/{lang}.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_text(key):
    """Return translated text from session state."""
    return st.session_state.translations.get(key, key)

def initialize_language():
    """Initialize Streamlit session with default English language."""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
        st.session_state.translations = load_translations('en')
