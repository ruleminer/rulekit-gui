import streamlit as st


def define_expert_preferred_extend(expert_preferred_conditions=None, expert_forbidden_conditions=None):
    expert_preferred_conditions = expert_preferred_conditions or []
    expert_forbidden_conditions = expert_forbidden_conditions or []

    def add_pref(condition_type, num, rule):
        tmp = ('preferred-' + condition_type + '-' + str(num - 1), rule)
        expert_preferred_conditions.append(tmp)

    def add_forb(condition_type, num, rule):
        tmp = ('forbidden-' + condition_type + '-' + str(num - 1), rule)
        expert_forbidden_conditions.append(tmp)

    st.write("")
    st.write("Preferred rules")
    col1, col2, col3 = st.columns([3, 2, 5])
    with col1:
        type_pref = st.selectbox(
            "Insert type of conditions", ('attribute', 'condition'), key="type_pref")
    with col2:
        num_pref = st.number_input(
            "Insert rule number", min_value=1, value=1, format='%i', key="num_pref")
    with col3:
        pref = st.text_input("Insert expert rules", value="", key="pref_txt")

    if pref != "" and pref != st.session_state["pref"]:
        add_pref(type_pref, num_pref, pref)
        st.session_state.pref = pref
    expert_preferred_conditions = st.data_editor(
        expert_preferred_conditions, num_rows="dynamic", width=1500, key="df1")

    st.write("")
    st.write("Forbidden rules")
    col1_forb, col2_forb, col3_forb = st.columns([3, 2, 5])
    with col1_forb:
        type_forb = st.selectbox(
            "Insert type of conditions", ('attribute', 'condition'), key="type_forb")
    with col2_forb:
        num_forb = st.number_input(
            "Insert rule number", min_value=1, value=1, format='%i', key="num_forb")
    with col3_forb:
        forb = st.text_input("Insert forbidden conditions",
                             value="", key="forb_txt")

    if forb != "" and forb != st.session_state["forb"]:
        add_forb(type_forb, num_forb, forb)
        st.session_state.forb = forb
    expert_forbidden_conditions = st.data_editor(
        expert_forbidden_conditions, num_rows="dynamic", width=1500, key="forb_df")

    return expert_preferred_conditions, expert_forbidden_conditions


def define_expert_preferred_induction(expert_rules=None):
    expert_rules = expert_rules or []

    def add_expert_rule(num, rule):
        tmp = ('rule-'+str(num-1), rule)
        expert_rules.append(tmp)

    st.write("")
    st.write("Expert induction rules")
    col1, col2 = st.columns([2, 5])
    with col1:
        num_exp = st.number_input(
            "Insert rule number", min_value=1, value=1, format='%i')
    with col2:
        ind_exp = st.text_input("Insert expert rules", value="")

    if ind_exp != "" and ind_exp != st.session_state["ind_exp"]:
        add_expert_rule(num_exp, ind_exp)
        st.session_state.ind_exp = ind_exp
    expert_rules = st.data_editor(expert_rules, num_rows="dynamic", width=1500)

    return expert_rules
