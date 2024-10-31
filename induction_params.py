import streamlit as st


def get_common_params():
    dictionary = {
        "max_rule_count": st.number_input("Max rule count", min_value=0, value=0, format='%i'),
        "enable_pruning": st.toggle("Enable pruning", value=True),
        "max_growing": st.number_input("Max growing", min_value=0, value=0, format='%i'),
        "minsupp_new": st.number_input("Minimum number of previously uncovered examples", min_value=0, value=5, format='%i'),
        "max_uncovered_fraction": st.number_input("Maximum fraction of uncovered examples", min_value=0.0, value=0.0, max_value=1.0, format='%f'),
        "ignore_missing": st.toggle("Ignore missing values", value=False),
        "select_best_candidate": st.toggle("Select best candidate", value=False),
        "complementary_conditions": st.toggle("Complementary conditions", value=False)
    }
    return dictionary


def get_common_params_expert():
    dictionary = {
        "extend_using_preferred": st.toggle("Extend using preferred", value=False),
        "extend_using_automatic": st.toggle("Extend using automatic", value=False),
        "induce_using_preferred": st.toggle("Induce using preferred", value=False),
        "induce_using_automatic": st.toggle("Induce using automatic", value=False),
        "preferred_conditions_per_rule": st.number_input("Preferred conditions per rule", min_value=0, value=0, format='%i'),
        "preferred_attributes_per_rule": st.number_input("Preferred attributes per rule", min_value=0, value=0, format='%i'),
    }
    return dictionary


def get_classification_params():
    dictionary = {
        "control_apriori_precision": st.toggle("Control apriori precision", value=True),
        "approximate_induction": st.toggle("Approximate induction", value=False),
        "approximate_bins_count": st.number_input("Approximate bins count", min_value=10, value=100, format='%i')
    }
    return dictionary


def get_regression_params():
    dictionary = {
        "mean_based_regression": st.toggle(
            "Mean based regression", value=True)
    }
    return dictionary
