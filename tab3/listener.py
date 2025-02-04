from typing import Optional

import pandas as pd
import streamlit as st
from rulekit.events import RuleInductionProgressListener
from rulekit.rules import BaseRule

from common.choices import EvaluationType


class MyProgressListener(RuleInductionProgressListener):
    """
    Progress listener compliant with RuleKit.
    Together with streamlit components, it displays the progress of ongoing rule induction.
    """

    def __init__(self, eval_type: EvaluationType, n_folds: Optional[int] = None):
        super().__init__()
        self.eval_type = eval_type
        self.progress_bar = st.progress(0)
        self.placeholder = st.empty()
        self.n_runs = 0
        self.n_folds = n_folds
        self.rule = None
        self._show_header()

    def on_new_rule(self, rule: BaseRule):
        self.rule = str(rule)

    def on_progress(
        self,
        total_examples_count: int,
        uncovered_examples_count: int,
    ):
        progress = (
            (total_examples_count - uncovered_examples_count) / total_examples_count)

        if progress > st.session_state.prev_progress:
            self._show_header(progress)
            st.session_state.prev_progress = progress

        if self.n_runs == 0:
            st.session_state.rules.append(self.rule)
            df = pd.Series(st.session_state.rules, name="Rules")
            self.placeholder.table(df)

    def finish(self):
        self.progress_bar.progress(100)
        self.progress_bar.empty()
        st.session_state.prev_progress = 0
        self.n_runs += 1

    def empty(self):
        self.placeholder.empty()

    def _show_header(self, progress: float = 0):
        if self.n_folds is not None:
            if self.n_runs > 0:
                text = f"Generating rules - iteration {self.n_runs}..."
            else:
                text = "Generating rules for the entire dataset..."
        else:
            text = "Generating rules..."
        self.progress_bar.progress(progress, text=text)
