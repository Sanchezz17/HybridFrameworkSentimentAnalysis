from typing import List
from random import random

crossover_probability = 0.5


def uniform_crossover(chromosome1: List[bool], chromosome2: List[bool]) -> (List[bool], List[bool]):
    result1 = []
    result2 = []
    chromosome_length = len(chromosome1)
    for i in range(chromosome_length):
        gene_from_1 = chromosome1[i]
        gene_from_2 = chromosome2[i]
        if random() < crossover_probability:
            result1.append(gene_from_1)
            result2.append(gene_from_2)
        else:
            result1.append(gene_from_2)
            result2.append(gene_from_1)
    return result1, result2
