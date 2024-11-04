import streamlit as st


def on_click_button_rule():
    st.session_state.button_rule = not st.session_state.button_rule
    st.session_state.gn = False
    st.session_state.click_stop = False
    st.session_state.prev_progress = 0


def on_click_gn():
    st.session_state.prev_progress = 0
    st.session_state.gn = True
    st.session_state.click_stop = False
