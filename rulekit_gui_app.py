import numpy as np
import pandas as pd
import streamlit as st
from rulekit.classification import ExpertRuleClassifier
from rulekit.classification import RuleClassifier
from rulekit.survival import ExpertSurvivalRules
from rulekit.survival import SurvivalRules
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from choices import DivType
from choices import EvaluationType
from choices import ModelType
from click_actions import on_click_button_rule
from click_actions import on_click_gn
from const import MEASURE_SELECTION
from evaluation import get_prediction_metrics
from evaluation import get_regression_metrics
from evaluation import get_ruleset_stats_class
from evaluation import get_ruleset_stats_reg
from evaluation import get_ruleset_stats_surv
from helpers import load_data
from listener import MyProgressListener
from models import define_model_classification
from models import define_model_regression
from models import define_model_survival
from session import set_session_state
from texts import DATASET_UPLOAD

st.set_page_config(page_title="RuleKit", initial_sidebar_state="expanded")

###   the beginning of the proper application code  ###
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model", "Rules", "Evaluation"])

## Defining session variables - they do not reset when the stop button is pressed. ##
set_session_state(st.session_state)

## Load a dataset that complies with the given conditions and in .csv format. ##
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


## If the data has been loaded then here are the defined objects to specify the model and its parameters ##
if st.session_state.data:
    with tab2:
        st.title("Model and Parameters")

        # Model type #
        genre = st.radio(
            "Model type",
            ModelType.choices(),
            index=0,
        )
        st.write("")

        # Evaluation type #
        eval_choices = EvaluationType.choices()
        if genre == ModelType.REGRESSION:
            eval_choices.remove(EvaluationType.CROSS_VALIDATION)
        eval_type = st.radio(
            "Evaluation parameters", eval_choices, index=0)

        # Definition of the independent variable and dependent variables #
        if genre == ModelType.CLASSIFICATION:
            x = data.drop(['target'], axis=1)
            y = data['target'].astype('category')
        elif genre == ModelType.REGRESSION:
            x = data.drop(['target'], axis=1)
            y = data['target']
        else:
            x = data.drop(['survival_status'], axis=1)
            y = data['survival_status']

        # Evaluation parameters - next step #
        if eval_type == EvaluationType.TRAIN_TEST:
            per_div = st.number_input(
                "Insert a percentage of the test set", value=0.20)

            if genre == ModelType.CLASSIFICATION:
                div_type = DivType.STRATIFIED
            else:
                div_type = st.radio(
                    "Hold out type", DivType.choices(), index=1)

        elif eval_type == EvaluationType.CROSS_VALIDATION:
            nfold = st.number_input("Insert a number of folds", value=5)

        st.write("")

        # Definition of the rule induction parameters #
        st.button('Define the induction parameters',
                  on_click=on_click_button_rule)

        if not st.session_state.button_rule:
            st.write("")
        else:
            st.write("")
            st.write("Algorithm parameters")

            # Definition of the model creation with the call of functions that allow the selection of parameters of the algorithm #
            if genre == ModelType.CLASSIFICATION:
                clf, metric, on_expert = define_model_classification()
            elif genre == ModelType.REGRESSION:
                clf, metric, on_expert = define_model_regression()
            else:
                clf, on_expert = define_model_survival()

    with tab3:

        # Division of the dataset into training and testing sets depending on chosen evaluation type #
        if st.session_state.data:
            if eval_type == EvaluationType.ONLY_TRAINING:
                x_train = x
                y_train = y

            elif eval_type == EvaluationType.TRAIN_TEST:
                if div_type == DivType.BY_ORDER:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x, y, test_size=per_div, shuffle=False)
                elif div_type == DivType.RANDOM:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x, y, test_size=per_div, shuffle=True)
                else:
                    x_train, x_test, y_train, y_test = train_test_split(
                        x, y, test_size=per_div, shuffle=True, stratify=y)

            else:
                skf = StratifiedKFold(n_splits=nfold)

        st.button("Generate Rules", on_click=on_click_gn)

        # Condition if dataset was loaded and the button was clicked #
        if st.session_state.gn and st.session_state.button_rule:

            # Model training process for EvaluationType.ONLY_TRAINING and EvaluationType.TRAIN_TEST evaluation types #
            if eval_type == EvaluationType.ONLY_TRAINING or eval_type == EvaluationType.TRAIN_TEST:
                nfold = 1

                # Definition of progress bar, stop button and placeholder -
                # this must be defined outside the class that follows the progress of rule induction #
                progress = 0
                progress_bar = st.progress(0)
                # st.button('Stop', on_click=click_stop)
                # placeholder is used to update the table with rules during the rule induction process
                placeholder = st.empty()

                # This is an exception to adding progress tracking for classification models.
                # This is because for this case when the progress bar is uploading the training process doesn't work. We are still working on it. #
                listener = MyProgressListener(
                    progress_bar, placeholder, eval_type)
                clf.add_event_listener(listener)

                # Model training process with updating progress bar and rule table #
                if on_expert:
                    clf.fit(x_train, y_train,
                            expert_preferred_conditions=st.session_state.pref_list,
                            expert_forbidden_conditions=st.session_state.forb_list,
                            expert_rules=st.session_state.ind_exp_list)
                else:
                    clf.fit(x_train, y_train)

                progress_bar.progress(100)
                progress_bar.empty()

                # Displaying the ruleset based on given model#
                ruleset = clf.model
                tmp = []
                for rule in ruleset.rules:
                    tmp.append({"Rules": str(rule)})
                placeholder.table(tmp)

            # Model training process for EvaluationType.CROSS_VALIDATION evaluation type #
            else:

                ruleset_stats = pd.DataFrame()
                prediction_metrics = pd.DataFrame()
                confusion_matrix_en = np.array([[0.0, 0.0], [0.0, 0.0]])
                survival_metrics = []

                entire_model = clf.fit(x, y)
                entire_ruleset = clf.model

                entire_ruleset_stats = []
                for rule in entire_ruleset.rules:
                    entire_ruleset_stats.append({"Rules": str(rule)})

                st.session_state_prev_progress = 0
                progress_bar = st.progress(0)
                placeholder = st.empty()
                for train_index, test_index in skf.split(x, y):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    if genre == ModelType.CLASSIFICATION:
                        if on_expert:
                            model_clone = ExpertRuleClassifier(
                                **entire_model.get_params())
                        else:
                            model_clone = RuleClassifier(
                                **entire_model.get_params())
                    elif genre == ModelType.SURVIVAL:
                        if on_expert:
                            model_clone = ExpertSurvivalRules(
                                **entire_model.get_params())
                        else:
                            model_clone = SurvivalRules(
                                **entire_model.get_params())

                    model_clone.add_event_listener(MyProgressListener(
                        progress_bar, placeholder, eval_type))
                    model_clone.fit(x_train, y_train)
                    ruleset = model_clone.model

                    # Obtaining the goodness of fit using a range of metrics - functions that were used are in the evaluation.py script. #

                    if genre == ModelType.CLASSIFICATION and st.session_state.button_rule:
                        measure = MEASURE_SELECTION.Metric[MEASURE_SELECTION.Desc == metric]
                        prediction, classification_metrics = model_clone.predict(
                            x_test, return_metrics=True)
                        tmp, confusion_matrix = get_prediction_metrics(
                            measure, prediction, y_test, classification_metrics)

                        prediction_metrics = pd.concat(
                            [prediction_metrics, tmp])
                        ruleset_stats = pd.concat([ruleset_stats,
                                                   get_ruleset_stats_class(measure, ruleset)])
                        confusion_matrix_en += confusion_matrix

                    elif genre == ModelType.SURVIVAL:
                        ibs = model_clone.score(x_test, y_test)
                        survival_metrics.append(ibs)
                        ruleset_stats = pd.concat(
                            [ruleset_stats, get_ruleset_stats_surv(ruleset)])

                progress_bar.progress(100)
                progress_bar.empty()

                ruleset_stats.index = [f"Fold {i}" for i in range(1, nfold+1)]
                st.write("Ruleset statistics")
                st.table(ruleset_stats)

                st.write("Rules for entire model")
                st.table(entire_ruleset_stats)

        else:
            if uploaded_file is None:
                st.write("")

            elif not st.session_state.button_rule:
                st.write("")

    with tab4:

        if st.session_state.data and st.session_state.button_rule and st.session_state.gn:
            # Obtaining the goodness of fit using a range of metrics - functions that were used are in the gui_function.py script. #
            # This is the same as in coss validation loop but for the EvaluationType.ONLY_TRAINING and EvaluationType.TRAIN_TEST evaluation types. #
            if eval_type == EvaluationType.ONLY_TRAINING:
                if genre == ModelType.CLASSIFICATION:
                    measure = MEASURE_SELECTION.Metric[MEASURE_SELECTION.Desc == metric]
                    prediction, model_metrics = clf.predict(
                        x_train, return_metrics=True)
                    new_model_metric, class_confusion_matrix = get_prediction_metrics(
                        measure, prediction, y_train, model_metrics)
                    ruleset_stats = get_ruleset_stats_class(measure, ruleset)
                    st.write("Confusion matrix")
                    st.dataframe(pd.DataFrame(class_confusion_matrix))
                elif genre == ModelType.REGRESSION:
                    measure = MEASURE_SELECTION.Metric[MEASURE_SELECTION.Desc == metric]
                    prediction = clf.predict(x_train)
                    new_model_metric = get_regression_metrics(
                        measure, prediction, y_train)
                    ruleset_stats = get_ruleset_stats_reg(measure, ruleset)
                else:
                    prediction = clf.predict(x_train)
                    ruleset_stats = get_ruleset_stats_surv(ruleset)

                if genre != ModelType.SURVIVAL:
                    new_model_metric.index = ["Values"]
                    st.write("Model statistics")
                    st.table(new_model_metric.transpose())

                ruleset_stats = pd.DataFrame(ruleset_stats)
                ruleset_stats.index = ["Values"]
                st.write("Ruleset statistics")
                st.table(ruleset_stats.transpose())

            elif eval_type == EvaluationType.TRAIN_TEST:
                if genre == ModelType.CLASSIFICATION:
                    measure = MEASURE_SELECTION.Metric[MEASURE_SELECTION.Desc == metric]
                    prediction, model_metrics = clf.predict(
                        x_test, return_metrics=True)
                    new_model_metric, class_confusion_matrix = get_prediction_metrics(
                        measure, prediction, y_test, model_metrics)
                    ruleset_stats = get_ruleset_stats_class(measure, ruleset)
                    st.write("Confusion matrix")
                    st.dataframe(pd.DataFrame(class_confusion_matrix))
                elif genre == ModelType.REGRESSION:
                    measure = MEASURE_SELECTION.Metric[MEASURE_SELECTION.Desc == metric]
                    prediction = clf.predict(x_test)
                    new_model_metric = get_regression_metrics(
                        measure, prediction, y_test.to_numpy())
                    ruleset_stats = get_ruleset_stats_reg(measure, ruleset)
                else:
                    prediction = clf.predict(x_test)
                    ruleset_stats = get_ruleset_stats_surv(ruleset)

                if genre != ModelType.SURVIVAL:
                    new_model_metric.index = ["Values"]
                    st.write("Model statistics")
                    st.table(new_model_metric.transpose())

                ruleset_stats = pd.DataFrame(ruleset_stats)
                ruleset_stats.index = ["Values"]
                st.write("Ruleset statistics")
                st.table(ruleset_stats.transpose())

            else:
                # Displaying the average ruleset statistics and prediction metrics based on models obtained in cross validation loop. #
                if genre == ModelType.CLASSIFICATION:
                    confusion_matrix_en /= nfold
                    st.write("Average confusion matrix")
                    st.dataframe(pd.DataFrame(confusion_matrix))
                    st.write("")
                    st.write("Average ruleset statistics")
                    st.table(ruleset_stats.mean())
                    st.write("")
                    st.write("Average prediction metrics")
                    st.table(prediction_metrics.mean())
                elif genre == ModelType.SURVIVAL:
                    st.write("Average survival metrics")
                    st.write(
                        f'Integrated Brier Score: {np.round(np.mean(survival_metrics), 6)}')
                    st.write("")
                    st.write("Average ruleset statistics")
                    st.table(ruleset_stats.mean())
