import pandas as pd


def format_table(table: pd.DataFrame):
    INTEGER_COLUMNS = {"rules count", "total conditions count",
                       "Covered by prediction", "Not covered by prediction"}
    table = table.T
    columns = set(table.columns)
    integer_columns = list(INTEGER_COLUMNS.intersection(columns))
    float_columns = list(columns - INTEGER_COLUMNS)
    table[integer_columns] = table[integer_columns].map(lambda x: f"{x:.0f}")
    table[float_columns] = table[float_columns].map(lambda x: f"{x:.3f}")
    return table.T
