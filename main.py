import pandas as pd
from sklearn.metrics import accuracy_score

from data_preparing import prepare_data
from text_cleaning import clean_text
# Датасет - обзоры фильмов на IMDB на английском языке
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
from lexicon_based_sentiment_analysis import calculate_polarity_score_swn
from text_preprocessing import preprocess_text

train_dataset = pd.read_csv('IMDB dataset/Train.csv')
test_dataset = pd.read_csv('IMDB dataset/Test.csv')

# Для быстрой проверки кода при разработке возьмем долю датасета в n процентов
n = 1
train_dataset = train_dataset.head(int(len(train_dataset) * (n / 100)))
test_dataset = test_dataset.head(int(len(test_dataset) * (n / 100)))

# Подготовка входных данных
train_dataset = prepare_data(train_dataset)
test_dataset = prepare_data(test_dataset)

# Подход 1: Анализ настроений на основе лексикона (Lexicon-based sentiment analysis)
y_pred_lexicon = test_dataset.keywords.apply(calculate_polarity_score_swn)
y_real = test_dataset.label
# Вычисляем accuracy - долю правильных ответов алгоритма
lexicon_accuracy = accuracy_score(y_real, y_pred_lexicon)
print(f"Lexicon-based accuracy: {lexicon_accuracy}")

# Подход 2: Анализ настроений на основе классификации и использования набора уникальных слов как признаков
# ML using bag of words as features
# toDo