import pandas as pd

from text_cleaning import clean_text
from text_preprocessing import preprocess_text


def prepare_data(dataset: pd.DataFrame) -> (pd.Series, pd.DataFrame, pd.Series):
    # Чистим текст от мусора, исправляем сленг, удаляем стоп-слова
    cleared_text = dataset.text.apply(clean_text)

    # Предварительная обработка: токенизация, стемминг, разметка частей речи
    # Инициализация списка ключевых слов текста и их свойств
    keywords = cleared_text.apply(preprocess_text)

    X = pd.DataFrame(keywords.to_list())
    print(X)

    y = dataset.label
    print(y)

    return keywords, X, y
