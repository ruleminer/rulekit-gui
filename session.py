import pandas as pd


DEFAULT_SESSION = {
    "prev_progress": 0,
    "pref_list": [],
    "forb_list": [],
    "expert_rules_list": [],
    "data": False,
    "ruleset_empty": False,
    "generation": False,
    "generated_rules": pd.Series(name="Rules"),
    "statistics": [],
    "indicators": [],
    "confusion_matrices": [],
}


def set_session_state(session_state):
    for key, value in DEFAULT_SESSION.items():
        if key not in session_state:
            setattr(session_state, key, value)
