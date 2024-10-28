import pandas as pd
import streamlit as st


def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    st.data_editor(
        data, num_rows="dynamic", disabled=False, hide_index=True, width=1500)
    st.session_state.data = True
    return data
