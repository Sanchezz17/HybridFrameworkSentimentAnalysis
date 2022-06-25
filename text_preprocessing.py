from collections import Counter
from typing import Dict

from nltk import PorterStemmer, pos_tag
from nltk.tokenize import word_tokenize

from text_keyword import TextKeyword

stemmer = PorterStemmer()


def preprocess_text(text: str) -> Dict[str, TextKeyword]:
    # Разбиваем текст на токены
    all_tokens = word_tokenize(text)
    tokens_counter = Counter(all_tokens)
    distinct_tokens = list(tokens_counter.keys())

    # Размечаем части речи токенов
    pos_tags = pos_tag(distinct_tokens)

    keywords = {}
    for token, pos in pos_tags:
        # Стемминг - получаем базовую форму слова
        stem = stemmer.stem(token)
        count = tokens_counter[token]
        keyword = TextKeyword(token=token, stem=stem, pos=pos, count=count)
        keywords[token] = keyword

    return keywords
