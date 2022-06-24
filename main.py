import pandas as pd

from data_cleaning import text_cleaning

# Датасет - обзоры фильмов на IMDB на английском языке
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
train_dataset = pd.read_csv('IMDB dataset/Train.csv')

# В разработке возьмем пока некую часть датасета, для быстрой проверки кода
n = 10
train_dataset = train_dataset.head(int(len(train_dataset) * (n / 100)))

print(train_dataset)

# Чистим текст
train_dataset.text = train_dataset.text.apply(text_cleaning)

print(train_dataset)


