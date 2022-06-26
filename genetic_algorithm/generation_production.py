import random
from typing import Dict, List

from genetic_algorithm.crossover import uniform_crossover
from genetic_algorithm.fitness_calculation import calculate_fitness
from genetic_algorithm.mutation import mutate
from text_keyword import TextKeyword


def produce_next_generation(keywords: Dict[str, TextKeyword],
                            sentence_keywords_counter: Dict[str, int],
                            sentence_label: int,
                            population: List[List[bool]]) -> List[List[bool]]:
    next_generation = []
    population_len = len(population)
    # Пока не наберем новое поколение
    while len(next_generation) < population_len:
        # Выбираем 4 случайных хромосомы из популяции
        i, j, k, l = random.sample(range(population_len), 4)
        ch1 = population[i]
        ch2 = population[j]
        ch3 = population[k]
        ch4 = population[l]

        # Считаем их фитнес-функцию
        fitness1 = calculate_fitness(keywords, sentence_keywords_counter, ch1, sentence_label)
        fitness2 = calculate_fitness(keywords, sentence_keywords_counter, ch2, sentence_label)
        fitness3 = calculate_fitness(keywords, sentence_keywords_counter, ch3, sentence_label)
        fitness4 = calculate_fitness(keywords, sentence_keywords_counter, ch4, sentence_label)

        # Выбираем лучшую хромосому среди первой и второй
        w1 = ch1 if fitness1 >= fitness2 else ch2

        # Выбираем лучшую хромосому среди третьей и четвертой
        w2 = ch3 if fitness3 >= fitness4 else ch4

        # Скрещиваем w1 и w2
        child1, child2 = uniform_crossover(w1, w2)

        # Производим мутацию child1 и child2
        mutate(child1, child2)

        # Добавляем в следующее поколение лучшую хромосому среди child1 и w1
        child1_fitness = calculate_fitness(keywords, sentence_keywords_counter, child1, sentence_label)
        w1_fitness = calculate_fitness(keywords, sentence_keywords_counter, w1, sentence_label)
        if child1_fitness > w1_fitness:
            next_generation.append(child1)
        else:
            next_generation.append(w1)

        # Добавляем в следующее поколение лучшую хромосому среди child2 и w2
        child2_fitness = calculate_fitness(keywords, sentence_keywords_counter, child1, sentence_label)
        w2_fitness = calculate_fitness(keywords, sentence_keywords_counter, w2, sentence_label)
        if child2_fitness > w2_fitness:
            next_generation.append(child2)
        else:
            next_generation.append(w2)

    return next_generation
