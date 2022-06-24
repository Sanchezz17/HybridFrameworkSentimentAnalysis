from typing import List

import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn

from text_keyword import TextKeyword

nltk.download('sentiwordnet')
nltk.download('wordnet')


def polarity_scoring_swn(keywords: List[TextKeyword]) -> (float, float, float):
    positive_score = 0
    negative_score = 0
    objective_score = 0
    for keyword in keywords:
        wordnet_tag = get_wordnet_tag(keyword.pos)
        if not wordnet_tag:
            continue
        swn_synsets = list(swn.senti_synsets(keyword.token, wordnet_tag))
        if not swn_synsets:
            continue
        synset = swn_synsets[0]
        if not synset:
            continue
        positive_score += synset.pos_score()
        negative_score += synset.neg_score()
        objective_score += synset.obj_score()
    return positive_score, negative_score, objective_score


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
