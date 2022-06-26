from typing import Dict, List, Set

import numpy as np

from genetic_algorithm.generation_production import produce_next_generation
from genetic_algorithm.population_initialization import initialize_population
from text_keyword import TextKeyword


def select_keywords_genetic_algorithm(keywords: Dict[str, TextKeyword],
                                      keywords_counters: List[Dict[str, int]],
                                      sentence_labels: np.ndarray,
                                      population_size: int,
                                      generations_count: int) -> List[str]:
    selected_keywords: Set[str] = set()
    for index, sentence_keywords_counter in enumerate(keywords_counters):
        sentence_label = sentence_labels[index]
        selected_keywords_for_sentence = select_keywords_for_sentence(keywords,
                                                                      sentence_keywords_counter,
                                                                      sentence_label,
                                                                      population_size,
                                                                      generations_count)
        selected_keywords = selected_keywords.union(selected_keywords_for_sentence)
    return list(selected_keywords)


def select_keywords_for_sentence(keywords: Dict[str, TextKeyword],
                                 sentence_keywords_counter: Dict[str, int],
                                 sentence_label: int,
                                 population_size: int,
                                 generations_count: int) -> Set[str]:
    chromosome_length = len(sentence_keywords_counter)
    population = initialize_population(population_size, chromosome_length)
    for i in range(generations_count):
        population = produce_next_generation(keywords,
                                             sentence_keywords_counter,
                                             sentence_label,
                                             population)
    best_chromosome = population[0]
    selected_keywords = {key for index, key in enumerate(sentence_keywords_counter)
                         if best_chromosome[index]}
    return selected_keywords
