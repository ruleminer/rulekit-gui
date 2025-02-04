import pandas as pd
import streamlit as st


def load_data():
    uploaded_file = st.file_uploader("File uploader")
    if uploaded_file is None:
        st.session_state.data = False
        st.write("")
        return None
    else:
        data = pd.read_csv(uploaded_file)
        data = st.data_editor(
            data, num_rows="dynamic", disabled=False, hide_index=True, width=1500)
        st.session_state.data = True
        return data
