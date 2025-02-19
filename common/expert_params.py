import pandas as pd
import streamlit as st


def get_common_expert_params():
    dictionary = {
        "extend_using_preferred": st.toggle("Add preferred conditions to existing rules", value=False),
        "extend_using_automatic": st.toggle("Add conditions to existing rules automatically", value=False),
        "induce_using_preferred": st.toggle("Allow induction of new rules that contain preferred conditions or are based on preferred attributes", value=False),
        "induce_using_automatic": st.toggle("Allow induction of new rules in a fully automatic manner", value=False),
        "preferred_conditions_per_rule": st.number_input("Maximum number of preferred conditions per rule", min_value=0, value=0, format="%i"),
        "preferred_attributes_per_rule": st.number_input("Maximum number of conditions built based on preferred attributes per rule", min_value=0, value=0, format="%i"),
    }
    return dictionary


def define_fit_expert_params():
    expert_params = {
        "expert_rules": [
            (f"expert-rule-{i + 1}", rule) for i, rule in enumerate(_define_expert_rules())
        ],
        "expert_preferred_conditions": [
            (f"preferred-{i + 1}", pref) for i, pref in enumerate(_define_preferred_elements())
        ],
        "expert_forbidden_conditions": [
            (f"forbidden-{i + 1}", forb) for i, forb in enumerate(_define_forbidden_elements())
        ],
    }
    st.write("For information on correct formatting of expert parameters, visit:")
    st.write("https://github.com/adaa-polsl/RuleKit/wiki/6-User-guided-induction")
    return expert_params


def _define_expert_rules():
    st.write("")
    st.write("Expert induction rules")
    expert_rules = pd.DataFrame(columns=["Expert rules"])
    expert_rules = st.data_editor(expert_rules, num_rows="dynamic", width=800)
    return expert_rules["Expert rules"].tolist()


def _define_preferred_elements():
    st.write("")
    st.write("Preferred attributes/conditions")
    preferred_elements = pd.DataFrame(
        columns=["Max occ.", "Preferred attributes/conditions"])
    column_config = {
        "Max occ.": {"width": 60},
        "Preferred attributes/conditions": {"width": 540},
    }
    preferred_elements = st.data_editor(
        preferred_elements, num_rows="dynamic", width=800, column_config=column_config)
    elements = preferred_elements.apply(
        lambda x: f"{x['Max occ.'] or 'inf'}: {x['Preferred attributes/conditions']}", axis=1)
    return elements.tolist()


def _define_forbidden_elements():
    st.write("")
    st.write("Forbidden attributes/conditions")
    forbidden_elements = pd.DataFrame(
        columns=["Forbidden attributes/conditions"])
    forbidden_elements = st.data_editor(
        forbidden_elements, num_rows="dynamic", width=1500)
    return forbidden_elements["Forbidden attributes/conditions"].tolist()
