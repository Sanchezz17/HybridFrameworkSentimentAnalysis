from typing import Dict

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

from text_keyword import TextKeyword

nltk.download('sentiwordnet')
nltk.download('wordnet')


def calculate_polarity_score_swn(keywords: Dict[str, TextKeyword]) -> int:
    polarity_score = 0
    for keyword in keywords.values():
        keyword_polarity_score = calculate_keyword_polarity_score(keyword)
        if not keyword_polarity_score:
            continue
        polarity_score += keyword_polarity_score
    return 1 if polarity_score > 0 else 0


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
    return (synset.pos_score() - synset.neg_score()) * keyword.count


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
