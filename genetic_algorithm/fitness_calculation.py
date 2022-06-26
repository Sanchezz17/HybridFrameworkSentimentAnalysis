from typing import List

from lexicon_based_sentiment_analysis import calculate_keyword_polarity_score
from text_keyword import TextKeyword


def calculate_fitness(keywords: List[TextKeyword], genotype: List[bool], sentiment_label: int):
    sum = 0
    scored_keywords_count = 0
    for index, keyword in keywords:
        if not genotype[index]:
            continue
        score = calculate_keyword_polarity_score(keyword)
        if not score:
            continue
        sum += score
        scored_keywords_count += 1
    sum /= scored_keywords_count
    return sentiment_label - sum
