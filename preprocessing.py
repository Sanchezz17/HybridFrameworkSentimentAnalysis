from typing import List

from nltk import PorterStemmer, pos_tag
from nltk.tokenize import word_tokenize

from text_keyword import TextKeyword

stemmer = PorterStemmer()


def text_preprocessing(text: str) -> List[TextKeyword]:
    # Разбиваем текст на токены
    tokens = word_tokenize(text)

    # Размечаем части речи токенов
    pos_tags = pos_tag(tokens)

    keywords = []
    for token, pos in pos_tags:
        # Стемминг - получаем базовую форму слова
        stem = stemmer.stem(token)
        keyword = TextKeyword(token=token, stem=stem, pos=pos)
        keywords.append(keyword)

    return keywords
