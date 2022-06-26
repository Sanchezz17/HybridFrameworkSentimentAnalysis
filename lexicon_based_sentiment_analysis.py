from typing import Dict, List
from functools import lru_cache

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

from text_keyword import TextKeyword

nltk.download('sentiwordnet')
nltk.download('wordnet')


def calculate_polarity_score_swn(keywords: Dict[str, TextKeyword],
                                 keywords_counters: List[Dict[str, int]]) -> List[int]:
    polarity_scores = []
    for keyword_counter in keywords_counters:
        polarity_score = 0
        for token, count in keyword_counter.items():
            keyword = keywords[token]
            keyword_polarity_score = calculate_keyword_polarity_score(keyword)
            if not keyword_polarity_score:
                continue
            polarity_score += keyword_polarity_score
        polarity_scores.append(1 if polarity_score > 0 else 0)
    return polarity_scores


@lru_cache(maxsize=None)
def calculate_keyword_polarity_score(keyword: TextKeyword) -> float | None:
    wordnet_tag = get_wordnet_tag(keyword.pos)
    if not wordnet_tag:
        return None
    swn_synsets = list(swn.senti_synsets(keyword.token, wordnet_tag))
    if not swn_synsets:
        return None
    synset = swn_synsets[0]
    if not synset:
        return None
    pos_score = synset.pos_score()
    neg_score = synset.neg_score()
    score = pos_score - neg_score
    return score


@lru_cache(maxsize=None)
def get_wordnet_tag(tag: str) -> str | None:
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None
