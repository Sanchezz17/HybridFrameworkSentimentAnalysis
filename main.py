import pandas as pd
from sklearn.metrics import accuracy_score

from data_preparing import prepare_data
# Датасет - обзоры фильмов на IMDB на английском языке
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
from lexicon_based_sentiment_analysis import calculate_polarity_score_swn

train_dataset = pd.read_csv('IMDB dataset/Train.csv')
test_dataset = pd.read_csv('IMDB dataset/Test.csv')

# Для быстрой проверки кода при разработке возьмем долю датасета в n процентов
n = 1
train_dataset = train_dataset.head(int(len(train_dataset) * (n / 100)))
test_dataset = test_dataset.head(int(len(test_dataset) * (n / 100)))

# Подготовка входных данных
keywords_train, X_train, y_train = prepare_data(train_dataset)
keywords_test, X_test, y_test = prepare_data(test_dataset)

# Подход 1: Анализ настроений на основе лексикона (Lexicon-based sentiment analysis)
# Проблема подхода: ограничен списком помеченных в SentiWordNet слов
y_pred_lexicon = keywords_test.apply(calculate_polarity_score_swn)
# Вычисляем accuracy - долю правильных ответов алгоритма
lexicon_accuracy = accuracy_score(y_test, y_pred_lexicon)
print(f"Lexicon-based accuracy: {lexicon_accuracy}")

# Подход 2: Анализ настроений на основе классификации и использования набора уникальных слов как признаков
# ML using bag of words as features
# Проблема подхода: слишком много признаков, не масштабируем на большие наборы данных
# toDo