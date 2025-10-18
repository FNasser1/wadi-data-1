import json
import streamlit as st
import os

def load_translations(lang):
    """Load language file (JSON) from locales/ directory."""
    try:
        # Try different possible paths for Streamlit Cloud
        possible_paths = [
            f'locales/{lang}.json',
            f'./locales/{lang}.json',
            f'/mount/src/wadi-data-1/locales/{lang}.json'
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except FileNotFoundError:
                continue
        
        # If no file found, return empty dict
        st.error(f"Translation file for {lang} not found")
        return {}
        
    except Exception as e:
        st.error(f"Error loading translations: {e}")
        return {}

def get_text(key):
    """Return translated text from session state."""
    return st.session_state.translations.get(key, key)

def initialize_language():
    """Initialize Streamlit session with default English language."""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
        st.session_state.translations = load_translations('en')
