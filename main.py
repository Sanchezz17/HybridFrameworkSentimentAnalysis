import pandas as pd
from sklearn.metrics import accuracy_score

from text_cleaning import clean_text
# Датасет - обзоры фильмов на IMDB на английском языке
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
from lexicon_based_sentiment_analysis import polarity_scoring_swn
from text_preprocessing import preprocess_text

train_dataset = pd.read_csv('IMDB dataset/Train.csv')
test_dataset = pd.read_csv('IMDB dataset/Test.csv')

# В разработке возьмем пока некую часть датасета, для быстрой проверки кода
n = 1
train_dataset = train_dataset.head(int(len(train_dataset) * (n / 100)))
test_dataset = test_dataset.head(int(len(test_dataset) * (n / 100)))

print(train_dataset)

# Чистим текст от мусора, исправляем сленг, удаляем стоп-слова
train_dataset.text = train_dataset.text.apply(clean_text)

print(train_dataset)

# Предварительная обработка: токенизация, стемминг, разметка частей речи
train_dataset = train_dataset.assign(keywords=train_dataset.text.apply(preprocess_text))

print(train_dataset)

# Lexicon-based sentiment analysis
y_pred_lexicon = train_dataset.keywords.apply(polarity_scoring_swn)
print(y_pred_lexicon)
y_real = train_dataset.label
# Вычисляем accuracy - долю правильных ответов алгоритма
lexicon_accuracy = accuracy_score(y_real, y_pred_lexicon)
print(f"Lexicon-based accuracy: {lexicon_accuracy}")
