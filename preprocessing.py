from typing import List

from text_keyword import TextKeyword

from nltk import PorterStemmer, pos_tag
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()


def text_preprocessing(text: str) -> List[TextKeyword]:
    # Разбиваем текст на токены
    tokens = word_tokenize(text)

    # Размечаем части речи токенов
    pos_tags = pos_tag(tokens)

    keywords = []
    for index, token in enumerate(tokens):
        # Стемминг - получаем базовую форму слова
        stem = stemmer.stem(token)
        pos = pos_tags[index]
        keyword = TextKeyword(token=token, stem=stem, pos=pos)
        keywords.append(keyword)

    return keywords
