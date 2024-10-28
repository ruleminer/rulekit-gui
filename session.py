DEFAULT_SESSION = {
    "prev_progress": 0,
    "pref": "none",
    "pref_list": [],
    "forb": "none",
    "forb_list": [],
    "ind_exp": "none",
    "ind_exp_list": [],
    "click_stop": False,
    "gn": False,
    "button_rule": True,
    "rule": [],
    "data": False,
}


def set_session_state(session_state):
    for key, value in DEFAULT_SESSION.items():
        if key not in session_state:
            setattr(session_state, key, value)
