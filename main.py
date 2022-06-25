import pandas as pd
from sklearn.metrics import accuracy_score

from data_preparing import prepare_data, get_x_y
from lexicon_based_sentiment_analysis import calculate_polarity_score_swn
from ml_using_bag_of_words_as_features import knn_classification

# Датасет - обзоры фильмов на IMDB на английском языке
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
train_dataset = pd.read_csv('IMDB dataset/Train.csv')
test_dataset = pd.read_csv('IMDB dataset/Test.csv')

# Для быстрой проверки кода при разработке возьмем долю датасета в n процентов
n = 5
train_dataset = train_dataset.head(int(len(train_dataset) * (n / 100)))
test_dataset = test_dataset.head(int(len(test_dataset) * (n / 100)))

# Подготовка входных данных
keywords_train, keywords_train_counters, keywords_dataframe_train = prepare_data(train_dataset)
X_train, y_train = get_x_y(train_dataset, keywords_dataframe_train)

keywords_test, keywords_test_counters, keywords_dataframe_test = prepare_data(test_dataset)
keywords_dataframe_test = keywords_dataframe_test.reindex(columns=keywords_dataframe_train.columns.tolist()).fillna(0)
X_test, y_test = get_x_y(test_dataset, keywords_dataframe_test)

# Подход 1: Анализ настроений на основе лексикона (Lexicon-based sentiment analysis)
# Проблема подхода: ограничен списком помеченных в SentiWordNet слов
y_pred_lexicon = calculate_polarity_score_swn(keywords_test, keywords_test_counters)
# Вычисляем accuracy - долю правильных ответов алгоритма
lexicon_accuracy = accuracy_score(y_test, y_pred_lexicon)
print(f"Lexicon-based accuracy: {lexicon_accuracy}")

# Подход 2: Анализ настроений на основе классификации и использования набора уникальных слов как признаков
# ML using bag of words as features
# Проблема подхода: слишком много признаков, не масштабируем на большие наборы данных
y_pred_ml = knn_classification(X_train, y_train, X_test)
# Вычисляем accuracy - долю правильных ответов алгоритма
ml_accuracy = accuracy_score(y_test, y_pred_ml)
print(f"ML using bag of words as features accuracy: {ml_accuracy}")
