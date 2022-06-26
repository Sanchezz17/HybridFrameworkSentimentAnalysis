from typing import List

import pandas as pd


def drop_columns_except(df: pd.DataFrame, except_columns: List[str]):
    return df.drop(df.columns.difference(except_columns), 1)