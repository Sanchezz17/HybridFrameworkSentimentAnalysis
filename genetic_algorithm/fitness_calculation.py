from typing import List, Dict

from lexicon_based_sentiment_analysis import calculate_keyword_polarity_score
from text_keyword import TextKeyword


def calculate_fitness(keywords: Dict[str, TextKeyword],
                      sentence_keywords_counter: Dict[str, int],
                      chromosome: List[bool],
                      sentence_label: int) -> float:
    scores_sum = 0
    for index, (token, count) in enumerate(sentence_keywords_counter.items()):
        if not chromosome[index]:
            continue
        keyword = keywords[token]
        # Оценка полярности слова в SentiWordNet
        score = calculate_keyword_polarity_score(keyword)
        if not score:
            continue
        scores_sum += score * count

    # Если текст негативный, то чем меньше сумма тем лучше
    if sentence_label == 0:
        scores_sum = -scores_sum

    return scores_sum
