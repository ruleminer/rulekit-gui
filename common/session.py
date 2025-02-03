import streamlit as st


DEFAULT_SESSION = {
    "data": False,
    "generation": False,
    "prev_progress": 0,
    "settings": {
        "model_type": None,
        "eval_type": None,
        "div_type": None,
        "per_div": None,
        "n_fold": None,
    },
    "x": None,
    "y": None,
    "train": [],
    "test": [],
    "statistics": [],
    "confusion_matrices": [],
    "indicators": [],
    "ruleset": None,
    "rules": [],
}


def set_session_state():
    for key, value in DEFAULT_SESSION.items():
        if key not in st.session_state:
            setattr(st.session_state, key, value)


def commit_current_model_settings(**kwargs):
    st.session_state.settings = kwargs


def reset_results_in_session():
    st.session_state.statistics = []
    st.session_state.confusion_matrices = []
    st.session_state.indicators = []
    st.session_state.rules = []
