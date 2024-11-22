import pandas as pd
import streamlit as st

from choices import ModelType


def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data = st.data_editor(
        data, num_rows="dynamic", disabled=False, hide_index=True, width=1500)
    st.session_state.data = True
    return data


def process_data(data: pd.DataFrame, problem_type: ModelType):
    try:
        if problem_type == ModelType.CLASSIFICATION:
            data = data.dropna(subset=['target'])
            x = data.drop(['target'], axis=1)
            # note: 'category' type of `y` leads to error in `decision-rules`
            # this is a temporary fix - it should be corrected in `decision-rules`
            y = data['target'].astype('str')
        elif problem_type == ModelType.REGRESSION:
            data = data.dropna(subset=['target'])
            x = data.drop(['target'], axis=1)
            y = data['target']
        else:
            data = data.dropna(subset=['survival_status'])
            x = data.drop(['survival_status'], axis=1)
            y = data['survival_status'].astype('int').astype('str')
        return x, y
    except KeyError:
        st.error(
            "The target column is not present in the dataset or a wrong model type was selected.")
        st.session_state.generation = False
        st.stop()
