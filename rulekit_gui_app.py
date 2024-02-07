import streamlit as st
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from rulekit.classification import RuleClassifier, ExpertRuleClassifier
from rulekit.regression import RuleRegressor, ExpertRuleRegressor
from rulekit.survival import SurvivalRules, ExpertSurvivalRules
from rulekit.params import Measures
from rulekit.events import RuleInductionProgressListener
from rulekit.rules import Rule
from gui_functions import *


st.set_page_config(page_title="RuleKit", initial_sidebar_state="expanded")
def load_data():
    data = pd.read_csv(uploaded_file)
    edited_df = st.data_editor(data, num_rows="dynamic", disabled=False, hide_index=True, width=1500)
    st.session_state.data = True
    return data


def metric_selection():
    met_ind_cluss = st.selectbox("Induction mearure", measure_selection, index=6)
    ind_cluss = measure_selection.Desc[measure_selection["Metric"] == met_ind_cluss].values[0]

    met_prun_cluss = st.selectbox("Pruning measure", measure_selection, index=6)
    prun_cluss = measure_selection.Desc[measure_selection["Metric"] == met_prun_cluss].values[0]

    met_vot_cluss = st.selectbox("Voting measure", measure_selection, index=6)
    vot_cluss = measure_selection.Desc[measure_selection["Metric"] == met_vot_cluss].values[0]

    dictionary = {"ind_cluss" : ind_cluss,
                    "prun_cluss" : prun_cluss,
                    "vot_cluss" : vot_cluss,}
    
    return dictionary



def common_params():
    dictionary = {
                "max_rule_count" : st.number_input("Max rule count", min_value=0, value=0, format = '%i'),
                "enable_pruning" : st.toggle("Enable pruning", value=True),
                "mgrowing" : st.number_input("Max growing", min_value=0, value=0, format = '%i'),
                "minsupp_new" : st.number_input("Minimum number of previously uncovered examples", min_value=0, value=5, format = '%i'),
                "max_uncovered_fraction" : st.number_input("Maximum fraction of uncovered examples", min_value=0.0, value=0.0, max_value=1.0, format = '%f'),
                "ignore_missing" : st.toggle("Ignore missing values", value=False),
                "select_best_candidate" : st.toggle("Select best candidate", value=False),
                "complementary_conditions" : st.toggle("Complementary conditions", value=False)}

    return dictionary


def classification_param():
    dictionary= {"control_apriori_precision" : st.toggle("Control apriori precision", value=True),
                 "approximate_induction" : st.toggle("Approximate induction", value=False),
                 "approximate_bins_count" : st.number_input("Approximate bins count", min_value=10, value=100, format = '%i')}
    return dictionary


def common_param_expert():
    dictionary = {"extend_using_preferred" : st.toggle("Extend using preferred", value=False),
                  "extend_using_automatic" : st.toggle("Extend using automatic", value=False),
                  "induce_using_preferred" : st.toggle("Induce using preferred", value=False),
                  "induce_using_automatic" : st.toggle("Induce using automatic", value=False),
                  #"preferred_conditions_per_rule" : st.number_input("Preferred conditions per rule", min_value=0, value=  int(1.796e+308), format = '%i'),
                  #"preferred_attributes_per_rule" : st.number_input("Preferred attributes per rule", min_value=0, value=  int(1.796e+308), format = '%i'),
                  }
    return dictionary


def regression_param():
    dictionary = {"mean_based_regression" : st.toggle("Mean based regression", value=True)}
    return dictionary


def define_params_class():
    
    on_expert = st.toggle('Do you want to performe expert induction?', value=False)

    metric = metric_selection()
    param = common_params()
    class_param = classification_param()
       

    if on_expert is False:
        clf = RuleClassifier(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            control_apriori_precision=class_param["control_apriori_precision"],
            approximate_bins_count=class_param["approximate_bins_count"],
            approximate_induction=class_param["approximate_induction"])
        
    elif on_expert:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = common_param_expert()

        clf = ExpertRuleClassifier(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            control_apriori_precision=class_param["control_apriori_precision"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            approximate_bins_count=class_param["approximate_bins_count"],
            approximate_induction=class_param["approximate_induction"],
            extend_using_preferred = expert_params["extend_using_preferred"],
            extend_using_automatic = expert_params["extend_using_automatic"],
            induce_using_preferred = expert_params["induce_using_preferred"],
            induce_using_automatic = expert_params["induce_using_automatic"],
            # consider_other_classes = st.toggle("Consider other classes", value=False),
            # preferred_conditions_per_rule = expert_params["preferred_conditions_per_rule"],
            # preferred_attributes_per_rule = expert_params["preferred_attributes_per_rule"]
            )
        
    return clf, metric["ind_cluss"]
        
    
def define_param_reg():
    
    on_expert = st.toggle('Do you want to performe expert induction?', value=False)

    metric = metric_selection()
    param = common_params()
    reg_param = regression_param()

    if on_expert is False:
        clf = RuleRegressor(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            mean_based_regression=reg_param["mean_based_regression"])
        
    elif on_expert:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = common_param_expert()

        clf = ExpertRuleRegressor(
            induction_measure=metric["ind_cluss"],
            pruning_measure=metric["prun_cluss"],
            voting_measure=metric["vot_cluss"],
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            mean_based_regression=reg_param["mean_based_regression"],
            extend_using_preferred = expert_params["extend_using_preferred"],
            extend_using_automatic = expert_params["extend_using_automatic"],
            induce_using_preferred = expert_params["induce_using_preferred"],
            induce_using_automatic = expert_params["induce_using_automatic"],
            # preferred_conditions_per_rule = expert_params["preferred_conditions_per_rule"],
            # preferred_attributes_per_rule = expert_params["preferred_attributes_per_rule"]
            )
        
    return clf, metric["ind_cluss"]

def define_param_surv():
    on_expert = st.toggle('Do you want to performe expert induction?', value=False)

    param = common_params()

    if on_expert is False:
        clf = SurvivalRules(
            survival_time_attr = 'survival_time',
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"])
    elif on_expert:
        st.text("")
        st.write("Expert induction parameters")
        expert_params = common_param_expert()

        clf = ExpertSurvivalRules(
            survival_time_attr = 'survival_time',
            max_rule_count=param["max_rule_count"],
            enable_pruning=param["enable_pruning"],
            max_growing=param["mgrowing"],
            minsupp_new=param["minsupp_new"],
            max_uncovered_fraction=param["max_uncovered_fraction"],
            ignore_missing=param['ignore_missing'],
            select_best_candidate=param["select_best_candidate"],
            complementary_conditions=param["complementary_conditions"],
            extend_using_preferred = expert_params["extend_using_preferred"],
            extend_using_automatic = expert_params["extend_using_automatic"],
            induce_using_preferred = expert_params["induce_using_preferred"],
            induce_using_automatic = expert_params["induce_using_automatic"],
            # preferred_conditions_per_rule = expert_params["preferred_conditions_per_rule"],
            # preferred_attributes_per_rule = expert_params["preferred_attributes_per_rule"]
            )
        
    return clf

    

measure_selection = pd.DataFrame({
    "Metric": ["Accuracy", "Binary Entropy", "C1", "C2", "C Foil", 
               "CN2 Significnce", "Correlation", "F Bayesian Confirmation", 
               "F Measure",  "FullCoverage", "GeoRSS", "GMeasure", 
               "InformationGain", "JMeasure", "Kappa", "Klosgen", "Laplace",
               "Lift", "LogicalSufficiency", "MEstimate", "MutualSupport",
               "Novelty", "OddsRatio", "OneWaySupport", "PawlakDependencyFactor",
               "Q2", "Precision", "RelativeRisk", "Ripper", "RuleInterest", 
               "RSS", "SBayesian", "Sensitivity", "Specificity", "TwoWaySupport",
               "WeightedLaplace", "WeightedRelativeAccuracy", "YAILS", "LogRank"],
    "Desc": [Measures.Accuracy, Measures.BinaryEntropy, Measures.C1, Measures.C2, Measures.CFoil, 
             Measures.CN2Significnce, Measures.Correlation, Measures.FBayesianConfirmation,
             Measures.FMeasure, Measures.FullCoverage, Measures.GeoRSS, Measures.GMeasure,
             Measures.InformationGain, Measures.JMeasure, Measures.Kappa, Measures.Klosgen, Measures.Laplace,
             Measures.Lift, Measures.LogicalSufficiency, Measures.MEstimate, Measures.MutualSupport,
             Measures.Novelty, Measures.OddsRatio, Measures.OneWaySupport, Measures.PawlakDependencyFactor,
             Measures.Q2, Measures.Precision, Measures.RelativeRisk, Measures.Ripper, Measures.RuleInterest,
             Measures.RSS, Measures.SBayesian, Measures.Sensitivity, Measures.Specificity, Measures.TwoWaySupport,
             Measures.WeightedLaplace, Measures.WeightedRelativeAccuracy, Measures.YAILS, Measures.LogRank]
})

class MyProgressListener(RuleInductionProgressListener):
    _uncovered_examples_count: int = None
    _should_stop = False
    df = []
    rule = 0

    def on_new_rule(self, rule: Rule):
        self.rule = str(rule)
        pass
        

    def on_progress(
        self,
        total_examples_count: int,
        uncovered_examples_count: int,
    ):
        # if uncovered_examples_count < total_examples_count * 0.1:
        #     self._should_stop = True
        
        # if st.session_state.click_stop:
        #     st.session_state.rule = self.df
        #     #st.write("Early stop")
        #     self._should_stop = True

        progress = ((total_examples_count - uncovered_examples_count)/total_examples_count)
        progress_bar.progress(progress, text = "Generating rules...")
        self.df.append(self.rule)
        placeholder.table(self.df)
        #self._uncovered_examples_count = uncovered_examples_count
    
    def should_stop(self) -> bool:
        return self._should_stop
    



###   the beginning of the proper application code  ###
tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model", "Rules", "Evaluation"])

## Defining session variables - they do not reset when the stop button is pressed. ##
if "click_stop" not in st.session_state:
    st.session_state.click_stop = False

if 'gn' not in st.session_state:
    st.session_state.gn = False

if 'button_rule' not in st.session_state:
    st.session_state.button_rule = False

if "rule" not in st.session_state:
    st.session_state.rule = []

if "data" not in st.session_state:
    st.session_state.data = False




## Load a dataset that complies with the given conditions and in .csv format. ##
with tab1:
    st.title("Dataset")
    st.write("Here will be the data set contruction rules that are necessary to run the application")
    uploaded_file = st.file_uploader('File uploader')
    
    if uploaded_file is None:
        st.write("")
    else:
        data = load_data()


## If the data has been loaded then here are the defined objects to specify the model and its parameters ##
if st.session_state.data:        
    with tab2:
        st.title("Model and Parameters")

        # Model type #
        genre = st.radio(
            "Model type",
            ["Classification", "Regresion", "Survival Analysis"],
            index=0,
            )
        st.write("")

        # Evaluation type #
        if genre == "Regresion":
            eval_type = st.radio("Evaluation parameters", 
                                ["Only training", "Training and testing - Hold out"], 
                                index=0
                                )
        else:
            eval_type = st.radio("Evaluation parameters", 
                                ["Only training", "Training and testing - Hold out", "Cross Validation"], 
                                index=0
                                )
            

        # Definition of the independent variable and dependent variables #
        if genre == "Classification":
            x = data.drop(['target'], axis=1)
            y = data['target'].astype('category')
        elif genre == "Regresion":
            x = data.drop(['target'], axis=1)
            y = data['target']
        elif genre == "Survival Analysis":
            x = data.drop(['survival_status'], axis=1)
            y = data['survival_status']


        # Evaluation parameters - next step #
        if eval_type == "Training and testing - Hold out":
            per_div = st.number_input("Insert a percentage of the test set", value=0.20)

            if genre == "Classification":
                div_type = "Stratified"
            else:
                div_type = st.radio("Hold out type", ["By order in dataset", "Random", "Stratified"], index=1)

        elif eval_type == "Cross Validation":
            nfold = st.number_input("Insert a number of folds", value=5)

        st.write("")



        # Definition of the rule induction parameters #
        # Function that change the state of session variable as response to the button click (Button - Define the induction parameters)
        def click_button_rule():
            st.session_state.button_rule = not st.session_state.button_rule
            st.session_state.gn = False
            st.session_state.click_stop = False

        st.button('Define the induction parameters', on_click=click_button_rule)

        if st.session_state.button_rule == False:
            st.write("")
        elif st.session_state.button_rule:
            st.write("")
            st.write("Algorithm parameters") 
            
            # Definition of the model creation with the call of functions that allow the selection of parameters of the algorithm #
            if genre == "Classification":
                clf, metric = define_params_class()
            elif genre == "Regresion":
                clf, metric = define_param_reg()
            elif genre == "Survival Analysis":
                clf = define_param_surv()
            



    with tab3:

        # Division of the dataset into training and testing sets depending of chosen evaluation type #
        if st.session_state.data:
            if eval_type == "Only training":
                    x_train = x
                    y_train = y

            elif eval_type == "Training and testing - Hold out":
                if div_type == "By order in dataset":
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=per_div, shuffle=False)
                elif div_type == "Random":
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=per_div, shuffle=True)
                elif div_type == "Stratified" or genre == "Classification":
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=per_div, shuffle=True, stratify=y)

            elif eval_type == "Cross Validation":
                skf = StratifiedKFold(n_splits=nfold)
        

        # Function that change the state of session variables as response to the button click (Button - Generate Rules) #
        def click_gn():
            st.session_state.gn = True
            st.session_state.click_stop = False

        st.button('Generate Rules', on_click=click_gn)

        # Condition if dataset was loaded and the button was clicked #
        if st.session_state.gn and st.session_state.button_rule:   

            # Model training process for "only training" and "training and testing - hold out" evaluation types #
            if eval_type == "Only training" or eval_type == "Training and testing - Hold out":

                # Function that change the state of session variable as response to the button click (Button - Stop). 
                # That button doesn't work at this moment, but still it is visible for user.#
                def click_stop():
                    st.session_state.click_stop = True
                    st.session_state.gn = True             
            
                # Definition of progress bar, stop button and placeholder - 
                # this must by defined outside the class that follows the progress of rule induction #
                progress_bar = st.progress(0)
                #st.button('Stop', on_click=click_stop)
                placeholder = st.empty()  # placceholder is used to updated the table with rules during the rule induction process


                # This is an exception to adding progress tracking for classification models.  
                # This is because for this case when the progress bar is uploading the trainig process doesn't work. We are sill working on it. #
                if genre != "Classification":
                    clf.add_event_listener(MyProgressListener())

                # Model training process with updating progress bar and rule table #
                clf.fit(x_train, y_train)
                progress_bar.progress(100)
                progress_bar.empty()                    

                # Displaying the ruleset based on given model#
                ruleset = clf.model
                tmp = []
                for rule in ruleset.rules:
                    tmp.append({"Rules" : str(rule)})
                placeholder.table(tmp)


            # Model training process for "cross validation" evaluation type #
            elif eval_type == "Cross Validation":
                
                ruleset_stats = pd.DataFrame()
                prediction_metrics = pd.DataFrame()
                confusion_matrix_en = np.array([[0.0, 0.0], [0.0, 0.0]])
                survival_metrics = []


                for train_index, test_index in skf.split(x, y):
                    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    clf.fit(x_train, y_train)
                    ruleset = clf.model

 
                    # Obtaining the godness of fit using a range of metrics - functions that were used are in the gui_function.py script. #
                    
                    if genre == "Classification" and st.session_state.button_rule:
                        measure = measure_selection.Metric[measure_selection.Desc == metric]
                        prediction, classification_metrics = clf.predict(x_test, return_metrics=True)
                        tmp, confusion_matrix = get_prediction_metrics(measure, prediction, y_test, classification_metrics)

                        prediction_metrics = pd.concat([prediction_metrics, tmp])
                        ruleset_stats = pd.concat([ruleset_stats, 
                                                get_ruleset_stats_class(measure, ruleset)])
                        confusion_matrix_en += confusion_matrix


                    elif genre == "Survival Analysis":
                        ibs = clf.score(x_test, y_test)
                        survival_metrics.append(ibs)
                        ruleset_stats = pd.concat([ruleset_stats, get_ruleset_stats_surv(ruleset)])
                

                # Displaying the average ruleset statistics and prediction metrics based on models obtained in cross validation loop. #
                if genre == "Classification":
                    confusion_matrix_en /= nfold
                    st.write("Average confusion matrix")
                    st.dataframe(pd.DataFrame(confusion_matrix))
                    st.write("")
                    st.write("Average ruleset statistics")
                    st.table(ruleset_stats.mean())
                    st.write("")
                    st.write("Average prediction metrics")
                    st.table(prediction_metrics.mean())
                elif genre == "Survival Analysis":
                    st.write("Average survival metrics")
                    st.write(f'Integrated Brier Score: {np.round(np.mean(survival_metrics), 6)}')
                    st.write("")
                    st.write("Average ruleset statistics")
                    st.table(ruleset_stats.mean())
                    

        else:
            if uploaded_file is None:
                st.write("") 
            
            elif st.session_state.button_rule == False:
                st.write("")

    with tab4:
            
            # Obtaining the godness of fit using a range of metrics - functions that were used are in the gui_function.py script. #
            # This is the same as in coss validation loop but for the "only training" and "training and testing - hold out" evaluation types. #
            if eval_type == "Only training": 
                if genre == "Classification" and st.session_state.button_rule:
                    measure = measure_selection.Metric[measure_selection.Desc == metric]
                    prediction, model_metrics = clf.predict(x_train, return_metrics=True)
                    new_model_metric, class_confusion_matrix = get_prediction_metrics(measure, prediction, y_train, model_metrics)
                    ruleset_stats = get_ruleset_stats_class(measure, ruleset)
                    st.write("Confusion matrix")
                    st.dataframe(pd.DataFrame(class_confusion_matrix))
                elif genre == "Regresion" and st.session_state.button_rule:
                    measure = measure_selection.Metric[measure_selection.Desc == metric]
                    prediction = clf.predict(x_train)
                    new_model_metric = get_regression_metrics(measure, prediction, y_train)
                    ruleset_stats = get_ruleset_stats_reg(measure, ruleset)
                elif genre == "Survival Analysis":
                    prediction = clf.predict(x_train)
                    ruleset_stats = get_ruleset_stats_surv(ruleset)

            elif eval_type == "Training and testing - Hold out":
                if genre == "Classification" and st.session_state.button_rule:
                    measure = measure_selection.Metric[measure_selection.Desc == metric]
                    prediction, model_metrics = clf.predict(x_test, return_metrics=True)
                    new_model_metric, class_confusion_matrix = get_prediction_metrics(measure, prediction, y_test, model_metrics)
                    ruleset_stats = get_ruleset_stats_class(measure, ruleset)
                    st.write("Confusion matrix")
                    st.dataframe(pd.DataFrame(class_confusion_matrix))
                elif genre == "Regresion" and st.session_state.button_rule:
                    measure = measure_selection.Metric[measure_selection.Desc == metric]
                    prediction = clf.predict(x_test)
                    new_model_metric = get_regression_metrics(measure, prediction, y_test)
                    ruleset_stats = get_ruleset_stats_reg(measure, ruleset)
                elif genre == "Survival Analysis":
                    prediction = clf.predict(x_test)
                    ruleset_stats = get_ruleset_stats_surv(ruleset)

            if genre != "Survival Analysis":
                st.write("Model statistics")
                st.table(new_model_metric)
            
            st.write("Ruleset statistics")
            st.table(ruleset_stats)
