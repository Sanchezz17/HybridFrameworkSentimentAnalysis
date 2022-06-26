from random import random, randrange
from typing import List

mutate_probability = 0.1


def mutate(chromosome1: List[bool], chromosome2: List[bool]) -> None:
    if random() < mutate_probability:
        k = randrange(len(chromosome1))
        chromosome1[k] = not chromosome1[k]
        k = randrange(len(chromosome2))
        chromosome2[k] = not chromosome2[k]
