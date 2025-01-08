DATASET_UPLOAD = """
In order to make the application work, your dataset must be properly prepared:
- the supported file format is CSV,
- UTF-8 encoding, the field separator is a comma, while the decimal is a period,
- missing values are represented as an empty character,
- the first line of the loaded file is the column names,
- the decision attribute must be named `target` in classification and regression, while for survival - `survival_time` and `survival_status`,
- in survival, `survival_status` must be a binary variable, with values {0, 1}.
"""

DESCRIPTION = """
Welcome to RuleKit GUI (v2.1.24).
<br>
For additional features and advanced functionality in rule-based analysis, please visit [RuleMiner](https://ruleminer.ai).
"""
