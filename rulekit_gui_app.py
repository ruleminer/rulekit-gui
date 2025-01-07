from statistics import calculate_prediction_indicators
from statistics import calculate_ruleset_stats

import pandas as pd
import streamlit as st
from decision_rules.ruleset_factories import ruleset_factory
from decision_rules.survival import SurvivalRuleSet
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from choices import DivType
from choices import EvaluationType
from choices import ModelType
from dataset import load_data
from dataset import process_data
from expert_params import parse_expert_params_to_fit
from helpers import format_confusion_matrix
from helpers import format_table
from helpers import get_mean_confusion_matrix
from helpers import get_mean_table
from helpers import toggle_generation
from listener import MyProgressListener
from models import define_model
from ruleset import display_ruleset
from ruleset import display_survival_ruleset
from session import set_session_state
from texts import DATASET_UPLOAD
from texts import DESCRIPTION

# Initialize the website and tabs
st.set_page_config(page_title="RuleKit", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)
with st.container(border=True):
    st.markdown(DESCRIPTION, unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model", "Rules", "Evaluation"])


# Define session variables - they do not reset when the stop button is pressed.
set_session_state(st.session_state)


# Load a dataset that complies with the given conditions and in .csv format.
with tab1:
    st.title("Dataset")
    with st.container(border=True):
        st.write(DATASET_UPLOAD)
    uploaded_file = st.file_uploader("File uploader")
    if uploaded_file is None:
        st.session_state.data = False
        st.write("")
    else:
        data = load_data(uploaded_file)


# If the data has been loaded then here are the defined objects to specify the model and its parameters
if st.session_state.data:
    with tab2:
        st.title("Model and Parameters")

        # Model type
        model_type = st.radio(
            "Model type",
            ModelType.choices(),
            index=0,
        )
        st.write("")

        # Evaluation type
        eval_choices = EvaluationType.choices()
        if model_type == ModelType.REGRESSION:
            eval_choices.remove(EvaluationType.CROSS_VALIDATION)
        eval_type = st.radio(
            "Evaluation parameters", eval_choices, index=0)

        # Define dataset split type
        nfold = None
        if eval_type == EvaluationType.TRAIN_TEST:
            per_div = st.number_input(
                "Insert a percentage of the test set", value=0.20)
            if model_type == ModelType.CLASSIFICATION:
                div_type = DivType.STRATIFIED
            else:
                div_type = st.radio(
                    "Hold out type", DivType.choices(), index=1)
        elif eval_type == EvaluationType.CROSS_VALIDATION:
            nfold = st.number_input("Insert a number of folds", value=5)

        st.write("")
        st.write("Algorithm parameters")
        clf, metric, on_expert = define_model(model_type)

    with tab3:
        # Proceed if the data has been loaded and rule generation has been initiated
        if not st.session_state.generation:
            st.button("Generate rules", on_click=toggle_generation)

        if st.session_state.generation:
            listener = MyProgressListener(
                eval_type, nfold)
            # Split the data into independent variables and dependent variable
            x, y = process_data(data, model_type)
            # Split the dataset according to settings
            if eval_type == EvaluationType.ONLY_TRAINING:
                x_train = x
                y_train = y
                x_test = x_train
                y_test = y_train
            elif eval_type == EvaluationType.TRAIN_TEST:
                shuffle = div_type in [DivType.RANDOM, DivType.STRATIFIED]
                stratify = y if div_type == DivType.STRATIFIED else None
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=per_div, shuffle=shuffle, stratify=stratify)
            else:
                skf = StratifiedKFold(n_splits=nfold)

            # Model training process for EvaluationType.ONLY_TRAINING and EvaluationType.TRAIN_TEST evaluation types
            if eval_type == EvaluationType.ONLY_TRAINING or eval_type == EvaluationType.TRAIN_TEST:
                nfold = 1
                progress = 0
                clf.add_event_listener(listener)

                # Model training process with updating progress bar and rule table
                try:
                    if on_expert:
                        clf.fit(x_train, y_train, **
                                parse_expert_params_to_fit())
                    else:
                        clf.fit(x_train, y_train)
                except Exception as e:
                    st.error(
                        "An error occurred during model training. Make sure the parameters are correct.")
                    st.stop()

                listener.finish()

                # Displaying the ruleset based on given model
                ruleset = clf.model
                for rule in clf.model.rules:
                    st.session_state.generated_rules[len(
                        st.session_state.generated_rules) + 1] = str(rule)

                if clf.model.rules:
                    ruleset = ruleset_factory(clf, x_train, y_train)
                    st.session_state.current_model = ruleset
                    if eval_type == EvaluationType.TRAIN_TEST:
                        train_ruleset_stats = calculate_ruleset_stats(
                            ruleset, x_train, y_train)
                        test_ruleset_stats = calculate_ruleset_stats(
                            ruleset, x_test, y_test)
                        train_prediction_indicators = calculate_prediction_indicators(
                            ruleset, x_train, y_train)
                        test_prediction_indicators = calculate_prediction_indicators(
                            ruleset, x_test, y_test)
                        if model_type == ModelType.CLASSIFICATION:
                            train_prediction_indicators, train_confusion_matrix = train_prediction_indicators
                            test_prediction_indicators, test_confusion_matrix = test_prediction_indicators
                            st.session_state.confusion_matrices.append(
                                train_confusion_matrix)
                            st.session_state.confusion_matrices.append(
                                test_confusion_matrix)
                        ruleset_stats = pd.DataFrame(
                            {"train": train_ruleset_stats["metric"],
                                "test": test_ruleset_stats["metric"]}
                        )
                        prediction_indicators = pd.DataFrame(
                            {"train": train_prediction_indicators["indicator"],
                                "test": test_prediction_indicators["indicator"]}
                        )
                    else:
                        ruleset_stats = calculate_ruleset_stats(
                            ruleset, x_test, y_test)
                        prediction_indicators = calculate_prediction_indicators(
                            ruleset, x_test, y_test)
                        if model_type == ModelType.CLASSIFICATION:
                            prediction_indicators, confusion_matrix = prediction_indicators
                            st.session_state.confusion_matrices.append(
                                confusion_matrix)
                    st.session_state.statistics.append(ruleset_stats)
                    st.session_state.indicators.append(prediction_indicators)
                    st.session_state.ruleset_empty = False
                else:
                    st.session_state.statistics = []
                    st.session_state.indicators = []
                    st.session_state.confusion_matrices = []
                    st.session_state.ruleset_empty = True
                    st.error(
                        "An empty ruleset was generated. Try changing the model settings.")

            # Model training process for EvaluationType.CROSS_VALIDATION evaluation type
            else:
                clf.add_event_listener(listener)
                try:
                    if on_expert:
                        entire_model = clf.fit(x, y, **
                                               parse_expert_params_to_fit())
                    else:
                        entire_model = clf.fit(x, y)
                except Exception as e:
                    st.error(
                        "An error occurred during model training. Make sure the parameters are correct.")
                    st.stop()
                listener.finish()

                if clf.model.rules:
                    for rule in clf.model.rules:
                        st.session_state.generated_rules[len(
                            st.session_state.generated_rules) + 1] = str(rule)
                    st.session_state.ruleset_empty = False
                    st.session_state.current_model = ruleset_factory(clf, x, y)
                else:
                    st.session_state.ruleset_empty = True

                st.session_state_prev_progress = 0
                if not st.session_state.ruleset_empty:
                    for train_index, test_index in skf.split(x, y):
                        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        try:
                            if on_expert:
                                clf.fit(x_train, y_train, **
                                        parse_expert_params_to_fit())
                            else:
                                clf.fit(x_train, y_train)
                        except Exception as e:
                            st.error(
                                "An error occurred during model training. Make sure the parameters are correct.")
                            st.stop()
                        listener.finish()

                        # Calculate ruleset statistics and prediction indicators for CV iteration
                        ruleset = ruleset_factory(clf, x_train, y_train)
                        iter_ruleset_stats = calculate_ruleset_stats(
                            ruleset, x_test, y_test)
                        if model_type == ModelType.CLASSIFICATION:
                            iter_prediction_indicators, confusion_matrix = calculate_prediction_indicators(
                                ruleset, x_test, y_test)
                            st.session_state.confusion_matrices.append(
                                confusion_matrix)
                        else:
                            iter_prediction_indicators = calculate_prediction_indicators(
                                ruleset, x_test, y_test)
                        st.session_state.statistics.append(iter_ruleset_stats)
                        st.session_state.indicators.append(
                            iter_prediction_indicators)
                else:
                    st.error(
                        "An empty ruleset was generated. Try changing the model settings.")

            st.session_state.generation = False
            st.session_state.previous_mode = eval_type
            st.session_state.previous_nfold = nfold

        if len(st.session_state.generated_rules) > 0:
            if st.session_state.previous_mode == EvaluationType.CROSS_VALIDATION:
                index = [f"Fold {i}" for i in range(
                    1, st.session_state.previous_nfold + 1)]
                ruleset_stats = pd.concat(st.session_state.statistics, axis=1)
                ruleset_stats.columns = index
                st.write("Ruleset statistics")
                st.table(format_table(ruleset_stats))
                st.write("Rules for entire model")
            else:
                st.write("Ruleset")
            if isinstance(st.session_state.current_model, SurvivalRuleSet):
                display_survival_ruleset(st.session_state.current_model)
            else:
                display_ruleset(st.session_state.current_model)

    with tab4:
        if not st.session_state.ruleset_empty and st.session_state.statistics:
            st.subheader("Ruleset statistics")
            ruleset_stats = get_mean_table(st.session_state.statistics)
            st.write(ruleset_stats.to_html(
                header=(eval_type == EvaluationType.TRAIN_TEST)), unsafe_allow_html=True)
            if model_type == ModelType.CLASSIFICATION and st.session_state.confusion_matrices:
                if eval_type == EvaluationType.TRAIN_TEST:
                    st.subheader("Confusion matrix (train data)")
                    st.write(
                        format_confusion_matrix(
                            st.session_state.confusion_matrices[0]).to_html(),
                        unsafe_allow_html=True,
                    )
                    st.subheader("Confusion matrix (test data)")
                    st.write(
                        format_confusion_matrix(
                            st.session_state.confusion_matrices[1]).to_html(),
                        unsafe_allow_html=True,
                    )
                else:
                    st.subheader("Confusion matrix")
                    confusion_matrix = get_mean_confusion_matrix(
                        st.session_state.confusion_matrices)
                    st.write(confusion_matrix.to_html(),
                             unsafe_allow_html=True)
            st.subheader("Prediction indicators")
            prediction_indicators = get_mean_table(st.session_state.indicators)
            st.write(prediction_indicators.to_html(
                header=(eval_type == EvaluationType.TRAIN_TEST)), unsafe_allow_html=True)
