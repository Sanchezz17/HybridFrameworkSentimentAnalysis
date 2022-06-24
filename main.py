import pandas as pd

from data_cleaning import text_cleaning

# Датасет - обзоры фильмов на IMDB на английском языке
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
from lexicon_based_sentiment_analysis import polarity_scoring_swn
from preprocessing import text_preprocessing

train_dataset = pd.read_csv('IMDB dataset/Train.csv')

# В разработке возьмем пока некую часть датасета, для быстрой проверки кода
n = 2
train_dataset = train_dataset.head(int(len(train_dataset) * (n / 100)))

print(train_dataset)

# Чистим текст от мусора, исправляем сленг, удаляем стоп-слова
train_dataset.text = train_dataset.text.apply(text_cleaning)

print(train_dataset)

# Предварительная обработка: токенизация, стемминг, разметка частей речи
train_dataset = train_dataset.assign(tokens=train_dataset.text.apply(text_preprocessing))

print(train_dataset)

# Lexicon-based sentiment analysis
train_dataset = train_dataset.assign(swn_scores=train_dataset.tokens.apply(polarity_scoring_swn))

print(train_dataset)