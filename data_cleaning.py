import re
import string
from functools import partial

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words("english")

# Инициализация словаря сленга
slang_map: dict[str] = {}
with open('slang.txt') as file:
    for line in file:
        abbreviation, decoding = line.partition("=")[::2]
        slang_map[abbreviation] = decoding

slang_words = sorted(slang_map, key=len, reverse=True)
slang_regex = re.compile(r"\b({})\b".format("|".join(map(re.escape, slang_words))))
replace_slang = partial(slang_regex.sub, lambda m: slang_map[m.group(1)])


def text_cleaning(text: str) -> str:
    # Приводим текст к нижнему регистру
    text = text.lower()

    # Заменяем сленг
    text = replace_slang(text)

    # Удаляем стоп-слова
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])

    # Удаляем не ascii символы
    text = text.encode('ascii', 'ignore').decode()

    # Удаляем URL
    text = re.sub(r'https*\S+', ' ', text)

    # Удаляем упоминания (mentions)
    text = re.sub(r'@\S+', ' ', text)

    # Удаляем хэштеги
    text = re.sub(r'#\S+', ' ', text)

    # Удаляем кавычки и следующий символ (Harper's -> Harper)
    text = re.sub(r'\'\w+', '', text)

    # Удаляем знаки препинания
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    # Удаляем числа
    text = re.sub(r'\w*\d+\w*', '', text)

    # Удаляем лишние пробелы
    text = re.sub(r'\s{2,}', ' ', text)

    return text
