import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder


def encoder(df: DataFrame, column: str, drop_nan: bool) -> DataFrame:
    if drop_nan:
        df = df.dropna(subset=[column])

    oe_style = OneHotEncoder()
    oe_results = oe_style.fit_transform(df[[column]])
    oe_results_array = oe_results.toarray()
    headers = oe_style.categories_

    def handle(x):
        t = np.array(list(map(lambda y: column + '_' + str(y), x)))
        return t

    df_encoded = pd.DataFrame(oe_results_array, columns=(list(map(lambda x: handle(x), headers))))
    df = df.drop(column, axis=1)
    df = df.join(df_encoded)

    return df


def flatten_cols(items: list, col_name: str) -> list:
    result = []
    for item in items:
        result.append(item[col_name])
    return result