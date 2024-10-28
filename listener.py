from typing import Optional

import streamlit as st
from rulekit.events import RuleInductionProgressListener
from rulekit.rules import Rule


class MyProgressListener(RuleInductionProgressListener):
    def __init__(self, progress_bar, placeholder, eval_type):
        self._progress_bar = progress_bar
        self._placeholder = placeholder
        self._eval_type = eval_type
        self._uncovered_examples_count: Optional[int] = None
        self._should_stop = False
        self.df = []
        self.rule = 0
        super().__init__()

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

        progress = (
            (total_examples_count - uncovered_examples_count) / total_examples_count)

        if progress > st.session_state.prev_progress:
            self._progress_bar.progress(progress, text="Generating rules...")
            st.session_state.prev_progress = progress

        self.df.append(self.rule)

        if self._eval_type != "Cross Validation":
            self._placeholder.table(self.df)
        self._uncovered_examples_count = uncovered_examples_count

    def should_stop(self) -> bool:
        return self._should_stop
