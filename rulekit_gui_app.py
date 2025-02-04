import streamlit as st

from common.choices import EvaluationType
from common.session import set_session_state
from common.texts import DATASET_UPLOAD
from common.texts import DESCRIPTION
from tab1 import load_data
from tab2 import define_model
from tab2 import get_training_settings
from tab3 import display_cross_validation
from tab3 import display_ruleset
from tab3 import toggle_generation
from tab3 import train_and_evaluate_all
from tab4 import display_results

"""
The main application consists of four tabs:
1. Dataset: Upload the dataset
2. Model: Choose the model and set the parameters
3. Rules: Generate the rules and display them
4. Evaluation: Display statistics and indicators describing the ruleset
"""

# Initialize page settings and tabs
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

# Set initial values in the session state
set_session_state()

with tab1:
    st.title("Dataset")
    with st.container(border=True):
        st.write(DATASET_UPLOAD)
    data = load_data()

# Only proceed with next steps if dataset has been uploaded
if st.session_state.data:
    with tab2:
        st.title("Model and Parameters")
        model_type, eval_type, div_type, per_div, n_fold = get_training_settings()

        st.write("")
        st.write("Algorithm parameters")
        clf, metric, on_expert = define_model(model_type)

    with tab3:
        if not st.session_state.generation:
            st.button("Generate rules", on_click=toggle_generation)

        if st.session_state.generation:
            train_and_evaluate_all(
                data, model_type, eval_type, div_type, per_div, n_fold, clf, on_expert
            )

        if st.session_state.ruleset is not None:
            if st.session_state.settings["eval_type"] == EvaluationType.CROSS_VALIDATION:
                display_cross_validation()
            else:
                st.write("Ruleset")
            display_ruleset()

    with tab4:
        if st.session_state.ruleset is not None:
            display_results()
