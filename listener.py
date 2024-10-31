from typing import Optional

import streamlit as st
from rulekit.events import RuleInductionProgressListener
from rulekit.rules import Rule

from choices import EvaluationType


class MyProgressListener(RuleInductionProgressListener):
    def __init__(self, eval_type: EvaluationType, nfolds: Optional[int] = None):
        super().__init__()
        self.eval_type = eval_type
        self.progress_bar = st.progress(0)
        self.placeholder = st.empty()
        self._uncovered_examples_count: Optional[int] = None
        self._should_stop = False
        self._nruns = 0
        self._nfolds = nfolds
        self.df = []
        self.rule = 0

    def on_new_rule(self, rule: Rule):
        self.rule = str(rule)

    def on_progress(
        self,
        total_examples_count: int,
        uncovered_examples_count: int,
    ):
        progress = (
            (total_examples_count - uncovered_examples_count) / total_examples_count)

        if progress > st.session_state.prev_progress:
            if self._nfolds is not None:
                if self._nruns > 0:
                    text = f"Generating rules - iteration {self._nruns}..."
                else:
                    text = "Generating rules for the entire dataset..."
            else:
                text = "Generating rules..."
            self.progress_bar.progress(progress, text=text)
            st.session_state.prev_progress = progress

        self.df.append(self.rule)

        if self.eval_type != "Cross Validation":
            self.placeholder.table(self.df)
        self._uncovered_examples_count = uncovered_examples_count

    def should_stop(self) -> bool:
        return self._should_stop

    def finish(self):
        self.progress_bar.progress(100)
        self.progress_bar.empty()
        st.session_state.prev_progress = 0
        self._nruns += 1
