DEFAULT_SESSION = {
    "prev_progress": 0,
    "pref_list": [],
    "forb_list": [],
    "expert_rules_list": [],
    "click_stop": False,
    "gn": False,
    "button_rule": True,
    "rule": [],
    "data": False,
    "ruleset_empty": False,
}


def set_session_state(session_state):
    for key, value in DEFAULT_SESSION.items():
        if key not in session_state:
            setattr(session_state, key, value)
