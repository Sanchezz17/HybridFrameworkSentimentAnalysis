import pandas as pd
from sklearn.metrics import accuracy_score

from data_preparing import prepare_data, get_x_y
from genetic_algorithm.features_selection import select_keywords_genetic_algorithm
from helpers import drop_columns_except
from lexicon_based_sentiment_analysis import calculate_polarity_score_swn
from ml_using_bag_of_words_as_features import classify_and_print_accuracy_all_methods

# Датасет - обзоры фильмов на IMDB на английском языке
# https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
train_dataset = pd.read_csv('IMDB dataset/Train.csv')
test_dataset = pd.read_csv('IMDB dataset/Test.csv')

# Для быстрой проверки кода при разработке возьмем долю датасета в n процентов
n = 1
train_dataset = train_dataset.head(int(len(train_dataset) * (n / 100)))
test_dataset = test_dataset.head(int(len(test_dataset) * (n / 100)))

# Подготовка входных данных
keywords_train, keywords_train_counters, keywords_dataframe_train = prepare_data(train_dataset)
X_train, y_train = get_x_y(train_dataset, keywords_dataframe_train)

keywords_test, keywords_test_counters, keywords_dataframe_test = prepare_data(test_dataset)
keywords_dataframe_test = keywords_dataframe_test.reindex(columns=keywords_dataframe_train.columns.tolist()).fillna(0)
X_test, y_test = get_x_y(test_dataset, keywords_dataframe_test)

# Для всех результатов вычисляем accuracy - долю правильных ответов алгоритма

# Подход 1: Анализ настроений на основе лексикона (Lexicon-based sentiment analysis)
# Проблема подхода: ограничен списком помеченных в SentiWordNet слов
y_pred_lexicon = calculate_polarity_score_swn(keywords_test, keywords_test_counters)
lexicon_accuracy = accuracy_score(y_test, y_pred_lexicon)
print(f"Lexicon-based accuracy: {lexicon_accuracy}\n")

# Подход 2: Анализ настроений на основе классификации и использования набора уникальных слов как признаков
# ML using bag of words as features
# Проблема подхода: слишком много признаков, не масштабируем на большие наборы данных
classify_and_print_accuracy_all_methods(X_train, y_train, X_test, y_test)

# Подход 3: Гибридный метод с оптимальным выбором признаков
# Оптимизируем список ключевых слов (которые явлются признаками),
# уменьшив его размер при сохранении точности
# c помощью генетического алгоритма
selected_keywords = select_keywords_genetic_algorithm(keywords=keywords_train,
                                                      keywords_counters=keywords_train_counters,
                                                      sentence_labels=y_test,
                                                      population_size=100,
                                                      generations_count=500)

# Оставим только отобранные ключевые слова (признаки)
X_train = drop_columns_except(X_train, except_columns=selected_keywords)
X_test = drop_columns_except(X_test, except_columns=selected_keywords)
print("\nAfter feature reduction\n")
classify_and_print_accuracy_all_methods(X_train, y_train, X_test, y_test)
