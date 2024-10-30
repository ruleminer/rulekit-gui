import pandas as pd
import streamlit as st

from choices import ModelType


def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    st.data_editor(
        data, num_rows="dynamic", disabled=False, hide_index=True, width=1500)
    st.session_state.data = True
    return data


def process_data(data: pd.DataFrame, problem_type: ModelType):
    if problem_type == ModelType.CLASSIFICATION:
        x = data.drop(['target'], axis=1)
        y = data['target'].astype('category')
    elif problem_type == ModelType.REGRESSION:
        x = data.drop(['target'], axis=1)
        y = data['target']
    else:
        x = data.drop(['survival_status'], axis=1)
        y = data['survival_status']
    return x, y
