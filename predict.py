import joblib
import numpy as np
import pandas as pd
import warnings
from matplotlib import pyplot as plt
from config import cols, primary_column
from utils import flatten_cols, encoder

plt.style.use('ggplot')  # определяем стиль
warnings.filterwarnings('ignore')  # исключение конфигурационных строчек (варнинги)

df = pd.read_csv('./datasets/dataset_predict.csv', encoding="utf-8", delimiter=";")

cols = flatten_cols(cols, 'name')

df_test = df[cols]
df_test = df_test.drop(primary_column, axis=1)
df_test = df_test.dropna()
df = df[cols].dropna()

for col in cols:
    if "need_encode" in col:
        df_test[col["name"]] = df_test[col["name"]].astype("string")
        df_test = encoder(df_test, col["name"], True)

model = joblib.load('model.sav')
result = model.predict(df_test.values)
r = pd.DataFrame(np.array(result), columns=['severity'])
g = pd.concat([df.reset_index(), r], axis=1)
g.to_csv('./predicted.csv', sep=';', encoding='UTF-8-SIG')
