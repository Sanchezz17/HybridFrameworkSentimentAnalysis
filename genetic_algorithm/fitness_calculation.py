from typing import List, Dict

from lexicon_based_sentiment_analysis import calculate_keyword_polarity_score
from text_keyword import TextKeyword


def calculate_fitness(keywords: Dict[str, TextKeyword],
                      sentence_keywords_counter: Dict[str, int],
                      chromosome: List[bool],
                      sentence_label: int) -> float:
    sum = 0
    scored_count = 0
    for index, (token, count) in enumerate(sentence_keywords_counter.values()):
        if not chromosome[index]:
            continue
        keyword = keywords[token]
        score = calculate_keyword_polarity_score(keyword)
        if not score:
            continue
        sum += score * count
        scored_count += count
    sum /= scored_count
    return sentence_label - sum
