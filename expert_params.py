import streamlit as st


def get_common_expert_params():
    dictionary = {
        "extend_using_preferred": st.toggle("Add preferred conditions to existing rules", value=False),
        "extend_using_automatic": st.toggle("Add conditions to existing rules automatically", value=False),
        "induce_using_preferred": st.toggle("Allow induction of new rules that contain preferred conditions or are based on preferred attributes", value=False),
        "induce_using_automatic": st.toggle("Allow induction of new rules in a fully automatic manner", value=False),
        "preferred_conditions_per_rule": st.number_input("Maximum number of preferred conditions per rule", min_value=0, value=0, format='%i'),
        "preferred_attributes_per_rule": st.number_input("Maximum number of conditions built based on preferred attributes per rule", min_value=0, value=0, format='%i'),
    }
    return dictionary


def parse_expert_params_to_fit():
    expert_params = {
        "expert_rules": [
            (f"expert_rule-{i+1}", rule) for i, rule in enumerate(st.session_state.expert_rules_list)
        ],
        "expert_preferred_conditions": [
            (f"preferred-{i+1}", f"1: {pref}") for i, pref in enumerate(st.session_state.pref_list)
        ],
        "expert_forbidden_conditions": [
            (f"forbidden-{i+1}", forb) for i, forb in enumerate(st.session_state.forb_list)
        ],
    }
    return expert_params


def define_fit_expert_params():
    st.session_state.expert_rules_list = _define_expert_rules(
        st.session_state.expert_rules_list)
    st.session_state.pref_list = _define_preferred_elements(
        st.session_state.pref_list)
    st.session_state.forb_list = _define_forbidden_elements(
        st.session_state.forb_list)


def _define_expert_rules(expert_rules=None):
    expert_rules = expert_rules or []

    st.write("")
    st.write("Expert induction rules")
    expert_rule = st.text_input(
        "Insert expert rule in the correct format", value="")

    if expert_rule != "" and expert_rule not in expert_rules:
        expert_rules.append(expert_rule)
    expert_rules = st.data_editor(
        expert_rules, num_rows="dynamic", width=1500, key="rules")

    return expert_rules


def _define_preferred_elements(preferred_elements=None):
    preferred_elements = preferred_elements or []

    st.write("")
    st.write("Preferred attributes/conditions")
    pref_elem = st.text_input(
        "Insert preferred attribute/condition in the correct format", value="", key="pref_elem_txt")
    if pref_elem != "" and pref_elem not in preferred_elements:
        preferred_elements.append(pref_elem)
    preferred_elements = st.data_editor(
        preferred_elements, num_rows="dynamic", width=1500, key="pref_elem")

    return preferred_elements


def _define_forbidden_elements(forbidden_elements=None):
    forbidden_elements = forbidden_elements or []

    st.write("")
    st.write("Forbidden attributes/conditions")
    pref_elem = st.text_input(
        "Insert forbidden attribute/condition in the correct format", value="", key="forb_elem_txt")
    if pref_elem != "" and pref_elem not in forbidden_elements:
        forbidden_elements.append(pref_elem)
    forbidden_elements = st.data_editor(
        forbidden_elements, num_rows="dynamic", width=1500, key="forb_elem")

    return forbidden_elements
