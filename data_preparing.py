from typing import Dict, List

import numpy as np
import pandas as pd

from text_keyword import TextKeyword
from text_preprocessing import preprocess_text


def prepare_data(dataset: pd.DataFrame) -> (Dict[str, TextKeyword], List[Dict[str, int]], pd.DataFrame):
    # Предварительная обработка: токенизация, стемминг, разметка частей речи
    # Инициализация списка ключевых слов текста и их свойств
    keywords, keywords_counters = preprocess_text(dataset.text)

    # Набор ключевых слов как набор признаков
    keywords_dataframe = pd.DataFrame(keywords_counters, dtype=pd.Int64Dtype())
    keywords_dataframe.fillna(0, inplace=True)

    return keywords, keywords_counters, keywords_dataframe


def get_x_y(dataset: pd.DataFrame, keywords_dataframe: pd.DataFrame) -> (np.ndarray, np.ndarray):
    # Входные данные для классификации
    X = keywords_dataframe.to_numpy()
    y = dataset.label.to_numpy()
    return X, y
