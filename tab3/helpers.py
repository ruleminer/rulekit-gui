import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from common.choices import DivType
from common.choices import EvaluationType


def toggle_generation():
    st.session_state.generation = True
    st.session_state.generated_rules = pd.Series(name="Rules")
    st.session_state.statistics = []
    st.session_state.indicators = []
    st.session_state.confusion_matrices = []


def make_splits(x, y, eval_type, div_type, per_div, n_fold):
    # Split the dataset according to settings
    match eval_type:
        case EvaluationType.ONLY_TRAINING:
            st.session_state.train = [(x, y)]
            st.session_state.test = [(x, y)]
        case EvaluationType.TRAIN_TEST:
            shuffle = div_type in [DivType.RANDOM, DivType.STRATIFIED]
            stratify = y if div_type == DivType.STRATIFIED else None
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=per_div, shuffle=shuffle, stratify=stratify)
            st.session_state.train = [(x_train, y_train)]
            st.session_state.test = [(x_test, y_test)]
        case EvaluationType.CROSS_VALIDATION:
            skf = StratifiedKFold(n_splits=n_fold)
            st.session_state.train = [(x, y)]
            st.session_state.test = [(x, y)]
            for train_index, test_index in skf.split(x, y):
                st.session_state.train.append(
                    (x.iloc[train_index], y.iloc[train_index]))
                st.session_state.test.append(
                    (x.iloc[test_index], y.iloc[test_index]))
