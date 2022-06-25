import pandas as pd

from text_cleaning import clean_text
from text_preprocessing import preprocess_text


def prepare_data(dataset: pd.DataFrame):
    # Чистим текст от мусора, исправляем сленг, удаляем стоп-слова
    dataset.text = dataset.text.apply(clean_text)

    # Предварительная обработка: токенизация, стемминг, разметка частей речи
    dataset = dataset.assign(keywords=dataset.text.apply(preprocess_text))
    print(dataset)

    return dataset
