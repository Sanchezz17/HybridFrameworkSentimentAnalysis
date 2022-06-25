from collections import Counter
from typing import Dict, List

import pandas as pd
from nltk import PorterStemmer, pos_tag
from nltk.tokenize import word_tokenize

from text_cleaning import clean_text
from text_keyword import TextKeyword

stemmer = PorterStemmer()


def preprocess_text(text_series: pd.Series) -> (Dict[str, TextKeyword], List[Dict[str, int]]):
    keywords = {}
    keywords_counters = []
    for _, text in text_series.iteritems():
        # Чистим текст от мусора, исправляем сленг, удаляем стоп-слова
        cleared_text = clean_text(text)

        # Разбиваем текст на токены
        all_tokens = word_tokenize(cleared_text)
        tokens_counter = Counter(all_tokens)
        distinct_tokens = list(tokens_counter.keys())

        # Размечаем части речи токенов
        pos_tags = pos_tag(distinct_tokens)

        for token, pos in pos_tags:
            if token in keywords:
                continue
            # Стемминг - получаем базовую форму слова
            stem = stemmer.stem(token)
            keyword = TextKeyword(token=token, stem=stem, pos=pos)
            keywords[token] = keyword
        keywords_counters.append(tokens_counter)
    return keywords, keywords_counters
